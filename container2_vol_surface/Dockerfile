# ===============================
# QuantStack GPU Worker (JAX)
# ===============================

FROM docker.io/nvidia/cuda:12.8.0-runtime-ubi9

# ---- OS + Python ----
RUN dnf -y install \
        python3 \
        python3-pip \
    && dnf clean all

# ---- Python deps (GPU) ----
RUN pip3 install --no-cache-dir \
        "jax[cuda12]" \
        dm-haiku \
        optax \
        numpy \
        orjson \
        python-dotenv \
        confluent-kafka \
        pyzmq

# ---- App ----
WORKDIR /app
COPY . /app

# ---- Default entrypoint ----
CMD ["python3", "surface_engine.py"]
