# Stage 1: Build
FROM python:3.9-slim as builder

# Hindari penulisan bytecode dan gunakan buffering nonaktif
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Instal dependency ke direktori /install
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Tambahkan PYTHONPATH agar Python dapat menemukan package yang diinstal
ENV PYTHONPATH=/install/lib/python3.9/site-packages

# Buat direktori NLTK agar resource dapat diunduh ke sana
RUN mkdir -p /root/nltk_data

# Pre-download resource NLTK ke direktori /root/nltk_data
RUN python -c "import nltk; \
    nltk.download('punkt_tab', download_dir='/root/nltk_data'); \
    nltk.download('stopwords', download_dir='/root/nltk_data'); \
    nltk.download('wordnet', download_dir='/root/nltk_data')"

# Stage 2: Final Image
FROM python:3.9-slim

WORKDIR /app

# Salin dependency yang telah diinstal dari builder
COPY --from=builder /install /usr/local

# Salin data NLTK yang telah di-download ke image final
COPY --from=builder /root/nltk_data /root/nltk_data

# Set environment variable agar NLTK dapat menemukan resource-nya
ENV NLTK_DATA=/root/nltk_data

# Salin seluruh kode aplikasi
COPY . .

EXPOSE 5001

ENV WORKERS=1

CMD ["sh", "-c", "gunicorn -w ${WORKERS:-2} -b 0.0.0.0:5001 app.embedder:app"]
