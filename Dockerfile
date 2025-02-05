# Gunakan Python 3.9
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy kode aplikasi ke dalam container
COPY . .

# Expose port untuk aplikasi
EXPOSE 5001

# Jalankan aplikasi menggunakan gunicorn
# CMD ["gunicorn", "-w", "${WORKERS:-2}", "-b", "0.0.0.0:5001", "app.embedder:app"]
CMD ["sh", "-c", "gunicorn -w ${WORKERS:-2} -b 0.0.0.0:5001 app.embedder:app"]