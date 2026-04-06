# ---- Base image ----
FROM python:3.11-slim

# ---- Environment sanity ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- System deps (minimal) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# ---- Workdir ----
WORKDIR /app

# ---- Python deps ----
# Keep this explicit and boring
RUN pip install --upgrade pip && \
    pip install \
        pyzmq \
        confluent-kafka \
        python-dotenv \
        schwabdev

# ---- App code ----
COPY . /app

# ---- Runtime defaults (override via compose/env) ----
ENV ZMQ_BIND=tcp://0.0.0.0:5555 \
    KAFKA_BOOTSTRAP=redpanda:9092 \
    TOKENS_DB=/data/tokens.db \
    DOTENV_PATH=/run/secrets/schwab.env

# ---- Entrypoint ----
CMD ["python", "schwab_ingest.py"]

