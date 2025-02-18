# Stage 1: Build
FROM python:3.9-slim as builder

# Hindari penulisan bytecode dan gunakan buffering nonaktif
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final Image
FROM python:3.9-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY . .

EXPOSE 5001

ENV WORKERS=1

CMD ["sh", "-c", "gunicorn -w ${WORKERS:-2} -b 0.0.0.0:5001 app.embedder:app"]