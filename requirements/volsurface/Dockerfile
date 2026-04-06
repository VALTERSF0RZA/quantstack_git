FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install GPU JAX (CUDA 12)
RUN pip install --no-cache-dir "jax[cuda12_pip]==0.4.30" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

COPY . .
CMD ["python", "service.py"]
