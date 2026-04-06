import os, time, json, logging, sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import zmq
from confluent_kafka import Producer
from dotenv import load_dotenv
import schwabdev

NY = ZoneInfo("America/New_York")

FIELD_MAP = {
    "1": "bid", "2": "ask", "3": "last", "4": "bid_size", "5": "ask_size",
    "6": "venue_or_flag", "7": "condition", "8": "seq"
}

# -------------------------
# Time formatting
# -------------------------
def fmt_utc(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(dt.microsecond/1000):03d}Z"

def fmt_ny(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).astimezone(NY)
    off = dt.strftime("%z")
    off = off[:3] + ":" + off[3:]
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(dt.microsecond/1000):03d}{off}"

# -------------------------
# Build 500 SPXW option keys
# (5 expiries × 50 strikes × C/P)
# -------------------------
def build_spxw_keys(center_strike: int, expiries: list[str]) -> str:
    strikes = [center_strike + i * 5 for i in range(-25, 25)]
    keys = []

    for exp in expiries:
        for k in strikes:
            keys.append(f".SPXW{exp}C{k}")
            keys.append(f".SPXW{exp}P{k}")

    if len(keys) != 500:
        raise RuntimeError(f"Expected 500 keys, got {len(keys)}")

    return ",".join(keys)

def main():
    logging.basicConfig(level=logging.INFO)

    load_dotenv(os.getenv("DOTENV_PATH", "/run/secrets/schwab.env"))
    tokens_db = os.getenv("TOKENS_DB", "/data/tokens.db")

    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP", "redpanda:9092")

    TOPIC_OPTIONS = "schwab.options.l1"
    TOPIC_SPOT    = "schwab.spot.l1"

    zmq_bind = os.getenv("ZMQ_BIND", "tcp://0.0.0.0:5555")

    client = schwabdev.Client(
        os.getenv("SCHWAB_CLIENT_ID"),
        os.getenv("SCHWAB_CLIENT_SECRET"),
        os.getenv("SCHWAB_REDIRECT_URI"),
        tokens_db=tokens_db,
    )
    streamer = schwabdev.Stream(client)

    # ZMQ (everything)
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(zmq_bind)

    # Kafka (selective)
    producer = Producer({"bootstrap.servers": kafka_bootstrap})

    state = {}
    last_emit = {}

    def emit_pretty(obj: dict):
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    # -------------------------
    # ROUTING LOGIC (FIXED)
    # -------------------------
    def publish(symbol: str, tick: dict):
        wire = json.dumps(tick, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

        # ZMQ gets EVERYTHING (ES / NQ / SPX / options)
        pub.send(wire)

        # Kafka is canonical only
        if symbol.startswith("."):
            # SPXW options
            producer.produce(
                TOPIC_OPTIONS,
                key=symbol.encode("utf-8"),
                value=wire,
            )

        elif symbol == "SPX":
            # SPX cash
            producer.produce(
                TOPIC_SPOT,
                key=b"SPX",
                value=wire,
            )

        # futures (/ES, /NQ) → ZMQ only

        producer.poll(0)

    def handler(message: str):
        try:
            envelope = json.loads(message)
        except Exception:
            return

        for block in envelope.get("data", []):
            service = block.get("service")
            ts_ms = block.get("timestamp")
            if not isinstance(ts_ms, int):
                continue

            for row in block.get("content", []):
                symbol = row.get("key")
                if not symbol:
                    continue

                st = state.setdefault(symbol, {})

                for k, v in row.items():
                    if k == "key":
                        continue
                    name = FIELD_MAP.get(str(k))
                    if name:
                        st[name] = v

                tick = {
                    "ts_ms": ts_ms,
                    "ts_utc": fmt_utc(ts_ms),
                    "ts_ny": fmt_ny(ts_ms),
                    "service": service,
                    "symbol": symbol,
                    "l1": {
                        "bid": st.get("bid"),
                        "ask": st.get("ask"),
                        "last": st.get("last"),
                        "bid_size": st.get("bid_size"),
                        "ask_size": st.get("ask_size"),
                    },
                    "meta": {
                        "condition": st.get("condition"),
                        "seq": st.get("seq"),
                    },
                }

                snap = (
                    tick["l1"]["bid"], tick["l1"]["ask"], tick["l1"]["last"],
                    tick["l1"]["bid_size"], tick["l1"]["ask_size"],
                    tick["meta"]["condition"], tick["meta"]["seq"]
                )
                if last_emit.get(symbol) == snap:
                    continue
                last_emit[symbol] = snap

                emit_pretty(tick)
                publish(symbol, tick)

    streamer.start(handler)
    time.sleep(1.0)

    # -------------------------
    # Subscriptions
    # -------------------------
    FIELDS = "0,1,2,3,4,5,6,7,8"

    # ZMQ visuals
    streamer.send(streamer.level_one_futures("/ES,/NQ", FIELDS))

    # Kafka canonical spot
    streamer.send(streamer.level_one_equities("SPX", FIELDS))

    # OPTIONS — moved forward to start 1/30
    center_strike = int(os.getenv("CENTER_STRIKE", "7000"))
    expiries = ["260130", "260131", "260202", "260206", "260220"]

    keys = build_spxw_keys(center_strike, expiries)
    logging.info(f"Subscribing SPXW options count=500 expiries={expiries}")
    streamer.send(streamer.level_one_options(keys, FIELDS))

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()

