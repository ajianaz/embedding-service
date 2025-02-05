# **Embedding Service** 🚀

A simple embedding API service built with **Flask** and **SentenceTransformers**, running on **Gunicorn**.
This service generates text embeddings using the `all-MiniLM-L6-v2` model.

---

## **📌 Features**

✅ **Fast API** powered by **Flask** and **Gunicorn**
✅ **Pre-trained Sentence Transformer** (`all-MiniLM-L6-v2`)
✅ **Dockerized & Available on GHCR**

---

## **🐳 Running with Docker**

### **1️⃣ Run Locally with Docker (Recommended)**

```sh
docker run -d -p 5001:5001 \
  --name embedding_service \
  -v .:/app \
  ghcr.io/ajianaz/embedding-service:latest
```

This will:
• Run the container in detached mode (-d)
• Expose port 5001:5001
• Mount the app folder as a volume for easy updates

Test the service:

```
curl -X POST http://localhost:5001/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

### **2️⃣ Pull Image from GHCR**

If you don’t want to build manually, you can pull the pre-built image:

```
docker pull ghcr.io/ajianaz/embedding-service:latest
```

Then run it using the same docker run command above.

### **3️⃣ Build & Run Locally (Optional)**

If you want to build your own image:

```
docker build -t embedding-service .
docker run -d -p 5001:5001 -v .:/app embedding-service
```

📡 Deploy with Docker Compose & Traefik

Create docker-compose.yml

Here’s a sample docker-compose.yml for embedding-service behind Traefik:

```
version: "3.8"

networks:
  traefik-network:
    external: true

services:
  embedding:
    image: ghcr.io/ajianaz/embedding-service:latest
    container_name: embedding_service
    environment:
      - WORKERS=2
    command: ["sh", "-c", "gunicorn -w ${WORKERS:-2} -b 0.0.0.0:5001 app.embedder:app"]
    networks:
      - traefik-network
    volumes:
      - .:/app  # Mount local app folder to the container
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.embedding.rule=Host(`YOUR-DOMAIN`)"
      - "traefik.http.routers.embedding.entrypoints=websecure"
      - "traefik.http.routers.embedding.tls.certresolver=myresolver"
      - "traefik.http.services.embedding.loadbalancer.server.port=5001"
```

**Run:**

```
docker-compose up -d
```

📨 API Usage

Endpoint: /embed

• Method: POST

• Request Body:

```
{
  "text": "Hello, world!"
}
```

• Response:

```
{
  "embedding": [0.123, -0.456, ...]
}
```

Test with curl:

```
curl -X POST http://localhost:5001/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

### ⚡ License

MIT License. Feel free to use and modify. 😊
