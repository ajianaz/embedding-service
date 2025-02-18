# ---------------------------
# Stage 1: Build Dependencies
# ---------------------------
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04 as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Pastikan requirements.txt mengandung paket GPU-enabled (misalnya, torch versi CUDA)
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------
# Stage 2: Final Image
# ---------------------------
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

WORKDIR /app

ENV PATH="/usr/local/bin:${PATH}"

COPY --from=builder /install /usr/local
COPY . .

EXPOSE 5001

ENV WORKERS=1

CMD ["sh", "-c", "gunicorn -w ${WORKERS:-2} -b 0.0.0.0:5001 app.embedder:app"]
