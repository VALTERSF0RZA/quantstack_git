import os, time, json, zmq, re
from datetime import datetime, timezone, date
from surface_engine import SurfaceEngine
from confluent_kafka import Consumer, Producer

# -------------------------
# Runtime config
# -------------------------
INPUT_MODE = os.getenv("INPUT_MODE", "kafka").lower()  # kafka | zmq

# ZMQ out (charts)
ZMQ_PUB_BIND = os.getenv("ZMQ_PUB_BIND", "tcp://0.0.0.0:5560")

# ZMQ in (only if INPUT_MODE=zmq)
ZMQ_SUB = os.getenv("ZMQ_SUB", "tcp://host.containers.internal:5555")
ZMQ_SUB_TOPIC = os.getenv("ZMQ_SUB_TOPIC", "")

# Kafka in (options ONLY)
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "redpanda:9092")
TOPIC_IN = os.getenv("TOPIC_IN", "schwab.options.l1")
GROUP_ID = os.getenv("GROUP_ID", "vol_surface")
AUTO_OFFSET_RESET = os.getenv("AUTO_OFFSET_RESET", "latest")

# NEW env: spot/underlying cache topic (so options IV has S_mid even if options ticks don't carry it)
TOPIC_SPOT = os.getenv("TOPIC_SPOT", "schwab.spot.l1")
SPOT_SYMBOL = os.getenv("SPOT_SYMBOL", "SPX")  # what symbol to accept for spot cache
SPOT_GROUP_ID = os.getenv("SPOT_GROUP_ID", f"{GROUP_ID}.spot")  # separate consumer group

# Kafka out (surface snapshots)
KAFKA_OUT = os.getenv("KAFKA_OUT", KAFKA_BROKER)
TOPIC_OUT = os.getenv("TOPIC_OUT", "vol.surface.snapshots")
PUBLISH_TO_KAFKA = os.getenv("PUBLISH_TO_KAFKA", "1") == "1"

PUBLISH_EVERY_SEC = float(os.getenv("PUBLISH_EVERY_SEC", "2.0"))
SYMBOL = os.getenv("SYMBOL", "SPX")
R = float(os.getenv("R", "0.0"))
Q = float(os.getenv("Q", "0.0"))

# T convention: ACT/365 (good enough for intraday surfaces; you can swap later)
DAY_COUNT = float(os.getenv("DAY_COUNT", "365.0"))  # 365.0 or 365.25
MIN_T_YEARS = float(os.getenv("MIN_T_YEARS", str(1.0 / 365.0 / 24.0)))  # ~1 hour floor

# Optional: if upstream does NOT provide underlying_mid, you can still run
# by feeding a constant fallback spot (not ideal, but useful for debugging).
UNDERLYING_FALLBACK = os.getenv("UNDERLYING_FALLBACK", "")  # e.g. "4900.0" or ""

# If you want to REQUIRE a live spot before processing options, set to "1"
REQUIRE_SPOT = os.getenv("REQUIRE_SPOT", "0") == "1"

# -------------------------
# (A) Regex parsing layer (feed decoding)
# -------------------------
# Matches .SPXW260129C7000  (root may vary; keep generic)
_OPT_RE = re.compile(r"^\.(?P<root>[A-Z]+W?)(?P<yymmdd>\d{6})(?P<cp>[CP])(?P<strike>\d+)$")

def _yymmdd_to_date(yymmdd: str) -> date:
    yy = int(yymmdd[0:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    # 2000-2099 window (fine for listed options)
    return date(2000 + yy, mm, dd)

def _expiry_to_T_years(expiry_yymmdd: str, now_utc: datetime) -> float:
    # simplest: expiry at end of day UTC; if you want exchange-close precision,
    # replace this with NY close logic (16:00 ET) for equities/index options.
    exp_d = _yymmdd_to_date(expiry_yymmdd)
    exp_dt = datetime(exp_d.year, exp_d.month, exp_d.day, 23, 59, 59, tzinfo=timezone.utc)

    dt_sec = (exp_dt - now_utc).total_seconds()
    if dt_sec <= 0:
        return 0.0
    T = dt_sec / (DAY_COUNT * 24.0 * 3600.0)
    return max(T, MIN_T_YEARS)

def parse_option_symbol(symbol: str):
    """
    Returns (K, T_years, cp) if symbol looks like .SPXW260129C7000, else None.
    """
    m = _OPT_RE.match(symbol or "")
    if not m:
        return None

    cp_char = m.group("cp")
    cp = +1 if cp_char == "C" else -1

    # strike in Schwab keys is typically integer strikes (SPXW increments are 5)
    K = float(m.group("strike"))

    now_utc = datetime.now(timezone.utc)
    T = _expiry_to_T_years(m.group("yymmdd"), now_utc)
    return (K, T, cp)

# -------------------------
# (A2) Spot normalization (cache underlying_mid)
# -------------------------
last_underlying_mid = {"v": None, "ts_ns": 0}

def normalize_spot(payload: dict):
    """
    Accepts either:
      - {"symbol": "SPX", "l1": {"bid":..., "ask":...}}
      - {"key": "SPX", "bid":..., "ask":...}
      - {"symbol": "SPX", "bid":..., "ask":...}
    Returns mid if matches SPOT_SYMBOL else None.
    """
    sym = payload.get("symbol") or payload.get("key")
    l1 = payload.get("l1", payload) if isinstance(payload.get("l1"), dict) else payload

    bid = l1.get("bid") or l1.get("bid_px")
    ask = l1.get("ask") or l1.get("ask_px")

    if sym != SPOT_SYMBOL or bid is None or ask is None:
        return None

    try:
        return 0.5 * (float(bid) + float(ask))
    except Exception:
        return None

# -------------------------
# (B) Normalization layer (enforce the SurfaceEngine contract)
# -------------------------
def normalize_event(payload: dict, spot_cache: float | None = None) -> dict:
    """
    Ensures we output:
      - underlying_mid (float)  [optional; can be None if not present and no cache/fallback]
      - K (float)
      - T (float, YEARS)
      - cp (+1/-1)
      - bid/ask
      - ts_recv_ns (int)
    """
    # --- timestamp ---
    ts_recv_ns = payload.get("ts_recv_ns") or payload.get("ts_ns")
    if ts_recv_ns is None:
        ts_recv_ns = int(time.time() * 1e9)
    else:
        ts_recv_ns = int(ts_recv_ns)

    # --- prices ---
    bid = payload.get("bid") or payload.get("bid_px")
    ask = payload.get("ask") or payload.get("ask_px")

    # Schwab L1 ticks often look like {"symbol": "...", "l1": {"bid":..., "ask":...}}
    if bid is None and isinstance(payload.get("l1"), dict):
        bid = payload["l1"].get("bid") or payload["l1"].get("bid_px")
    if ask is None and isinstance(payload.get("l1"), dict):
        ask = payload["l1"].get("ask") or payload["l1"].get("ask_px")

    # --- underlying (1) from payload ---
    underlying_mid = (
        payload.get("underlying_mid") or payload.get("S_mid")
        or payload.get("underlying") or payload.get("spot") or payload.get("S")
    )
    # If upstream nested it
    if underlying_mid is None and isinstance(payload.get("underlying"), dict):
        underlying_mid = payload["underlying"].get("mid")

    # --- underlying (2) from cache ---
    if underlying_mid is None:
        underlying_mid = spot_cache

    # --- underlying (3) from fallback (debug only) ---
    if underlying_mid is None and UNDERLYING_FALLBACK:
        try:
            underlying_mid = float(UNDERLYING_FALLBACK)
        except Exception:
            underlying_mid = None

    # --- option terms ---
    K = payload.get("K") or payload.get("strike")
    T = payload.get("T") or payload.get("T_years") or payload.get("ttm_years")
    cp = payload.get("cp") or payload.get("right") or payload.get("call_put")

    # If not provided, parse from symbol string
    symbol = payload.get("symbol") or payload.get("key")
    if (K is None or T is None or cp is None) and isinstance(symbol, str):
        parsed = parse_option_symbol(symbol)
        if parsed is not None:
            K2, T2, cp2 = parsed
            if K is None: K = K2
            if T is None: T = T2
            if cp is None: cp = cp2

    # cp normalization (string -> +/-1)
    if isinstance(cp, str):
        c = cp.upper()
        cp = 1 if c in ("C", "CALL") else (-1 if c in ("P", "PUT") else cp)

    # --- hard requirement checks for option quotes ---
    if K is None or T is None or cp is None or bid is None or ask is None:
        raise ValueError("not an option-quote event")

    # IMPORTANT: enforce YEARS
    # If someone accidentally passes "days" upstream, it will be huge (e.g. 7).
    # A crude guard: if T > 3, assume it's days and convert. (You can tighten this.)
    T = float(T)
    if T > 3.0:
        T = max(T / DAY_COUNT, MIN_T_YEARS)

    # If you require a live spot and we still don't have it, fail loudly.
    if REQUIRE_SPOT and underlying_mid is None:
        raise ValueError("missing underlying_mid (no spot cache yet)")

    return {
        "underlying_mid": float(underlying_mid) if underlying_mid is not None else None,
        "K": float(K),
        "T": float(T),          # YEARS
        "cp": int(cp),
        "bid": float(bid),
        "ask": float(ask),
        "ts_recv_ns": ts_recv_ns,
        "symbol": symbol,
    }

def _mk_consumer(group_id: str) -> Consumer:
    return Consumer({
        "bootstrap.servers": KAFKA_BROKER,
        "group.id": group_id,
        "auto.offset.reset": AUTO_OFFSET_RESET,
        "enable.auto.commit": True,
    })

def main():
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(ZMQ_PUB_BIND)

    producer = Producer({"bootstrap.servers": KAFKA_OUT})
    engine = SurfaceEngine(symbol=SYMBOL, r=R, q=Q)

    last_pub = 0.0
    good = bad = 0
    spot_good = spot_bad = 0

    if INPUT_MODE == "kafka":
        # Two consumers: one for options, one for spot cache.
        c_opt = _mk_consumer(GROUP_ID)
        c_spot = _mk_consumer(SPOT_GROUP_ID)

        c_opt.subscribe([TOPIC_IN])
        c_spot.subscribe([TOPIC_SPOT])

        print(
            f"[surface] INPUT=KAFKA opt_topic={TOPIC_IN} spot_topic={TOPIC_SPOT} spot_symbol={SPOT_SYMBOL} "
            f"| broker={KAFKA_BROKER} | ZMQ_OUT={ZMQ_PUB_BIND} | OUT={KAFKA_OUT}/{TOPIC_OUT} enabled={PUBLISH_TO_KAFKA} "
            f"| DAY_COUNT={DAY_COUNT} REQUIRE_SPOT={REQUIRE_SPOT}"
        )

        try:
            while True:
                # --- poll spot quickly (non-blocking) ---
                msg_spot = c_spot.poll(0.0)
                if msg_spot is not None:
                    if msg_spot.error():
                        spot_bad += 1
                    else:
                        try:
                            obj = json.loads(msg_spot.value().decode("utf-8"))
                            payload = obj.get("payload", obj)
                            s = normalize_spot(payload)
                            if s is not None:
                                last_underlying_mid["v"] = s
                                last_underlying_mid["ts_ns"] = int(time.time() * 1e9)
                                # Keep engine's underlying fresh too (optional)
                                engine.update_underlying(s, last_underlying_mid["ts_ns"])
                                spot_good += 1
                        except Exception:
                            spot_bad += 1

                # --- poll options (blocking up to 1s) ---
                msg = c_opt.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    bad += 1
                    continue

                try:
                    obj = json.loads(msg.value().decode("utf-8"))
                    payload = obj.get("payload", obj)

                    m = normalize_event(payload, spot_cache=last_underlying_mid["v"])

                    # Underlying: update only when present (or cache injected)
                    if m["underlying_mid"] is not None:
                        engine.update_underlying(m["underlying_mid"], m["ts_recv_ns"])

                    # Options only: upsert option terms (already normalized to YEARS)
                    engine.upsert_option(m["K"], m["T"], m["cp"], m["bid"], m["ask"], m["ts_recv_ns"])
                    good += 1
                except Exception:
                    bad += 1

                now = time.time()
                if now - last_pub >= PUBLISH_EVERY_SEC:
                    surf = engine.compute_surface_points()
                    if surf:
                        blob = json.dumps(surf).encode("utf-8")

                        # ZMQ OUT (charts)
                        pub.send_multipart([b"surface.iv.raw", blob])

                        # Kafka OUT (Redpanda bus)
                        if PUBLISH_TO_KAFKA:
                            producer.produce(TOPIC_OUT, value=blob)
                            producer.poll(0)

                        print(
                            f"[surface] published points={surf['count']} S={surf['S_mid']:.2f} "
                            f"opt_good={good} opt_bad={bad} spot_good={spot_good} spot_bad={spot_bad} "
                            f"spot_cache={'Y' if last_underlying_mid['v'] is not None else 'N'}"
                        )
                    last_pub = now
        finally:
            c_opt.close()
            c_spot.close()

    else:
        # ZMQ input mode: you can still use a spot cache, but that requires a second ZMQ sub,
        # or you rely on UNDERLYING_FALLBACK / underlying_mid present in the payload.
        sub = ctx.socket(zmq.SUB)
        sub.connect(ZMQ_SUB)
        sub.setsockopt_string(zmq.SUBSCRIBE, ZMQ_SUB_TOPIC)

        print(
            f"[surface] INPUT=ZMQ sub={ZMQ_SUB} | ZMQ_OUT={ZMQ_PUB_BIND} | "
            f"OUT={KAFKA_OUT}/{TOPIC_OUT} enabled={PUBLISH_TO_KAFKA} | "
            f"DAY_COUNT={DAY_COUNT} REQUIRE_SPOT={REQUIRE_SPOT}"
        )

        while True:
            raw = sub.recv()
            obj = json.loads(raw.decode("utf-8"))

            m = normalize_event(obj, spot_cache=last_underlying_mid["v"])
            if m["underlying_mid"] is not None:
                engine.update_underlying(m["underlying_mid"], m["ts_recv_ns"])
            engine.upsert_option(m["K"], m["T"], m["cp"], m["bid"], m["ask"], m["ts_recv_ns"])

            now = time.time()
            if now - last_pub >= PUBLISH_EVERY_SEC:
                surf = engine.compute_surface_points()
                if surf:
                    blob = json.dumps(surf).encode("utf-8")
                    pub.send_multipart([b"surface.iv.raw", blob])
                    if PUBLISH_TO_KAFKA:
                        producer.produce(TOPIC_OUT, value=blob)
                        producer.poll(0)
                    print(f"[surface] published points={surf['count']} S={surf['S_mid']:.2f} opt_good={good} opt_bad={bad}")
                last_pub = now

if __name__ == "__main__":
    main()

