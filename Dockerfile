# Gunakan Python 3.9
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy kode aplikasi ke dalam container
COPY app/ .

# Jalankan aplikasi
CMD ["python", "embedder.py"]