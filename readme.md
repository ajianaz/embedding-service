# Embedding Service 🚀

A simple embedding API service built with **Flask** and **SentenceTransformers**, running on **Gunicorn**.
This service generates text embeddings using the `all-MiniLM-L6-v2` model, and supports optional text chunking and saving to Qdrant.

---

## Features

- **Fast API** powered by **Flask** and **Gunicorn**
- Uses pre-trained **Sentence Transformer** (`all-MiniLM-L6-v2`)
- **Dockerized** & available on **GHCR**
- Supports both **single string** and **list of strings** as input
- Optional **text chunking** (controlled via parameter or environment variable)
- Optional saving of embeddings to **Qdrant** (controlled via parameter or environment variable)
- Provides endpoints for generating embeddings, searching embeddings, and listing available models
- **Bearer token** authentication via the `API_KEY` environment variable

---

## API Endpoints

### 1. List Models

- **Endpoint:** `GET /v1/models`
- **Description:** Returns a list of available models.
- **Example Response:**

  ```json
  {
    "object": "list",
    "data": [
      {
        "id": "all-MiniLM-L6-v2",
        "object": "model",
        "owned_by": "ajianaz-dev"
      }
    ]
  }
  ```

---

### 2. Generate Embeddings

- **Endpoint:** `POST /v1/embeddings`
- **Authentication:**
  Include header `Authorization: Bearer <API_KEY>`
- **Request Body Options:**

  - **`input`**: A string or a list of strings.
  - **`collection`**: (Optional) Name of the Qdrant collection. Default is set in the code.
  - **`metadata`**: (Optional) Additional metadata to be saved along with each embedding.
  - **`chunk`**: (Optional) Boolean (`true`/`false`) to enable text chunking.
    Default is taken from the environment variable `DEFAULT_CHUNK` (default is `false`).
  - **`chunk_size`**: (Optional) The size of each text chunk. Default is `256`.
  - **`overlap`**: (Optional) The number of overlapping words between chunks. Default is `50`.
  - **`save_to_qdrant`**: (Optional) Boolean (`true`/`false`) to save each embedding to Qdrant.
    Default is taken from the environment variable `DEFAULT_SAVE_QDRANT` (default is `false`).

- **Example Request:**

  ```bash
  curl -X POST http://localhost:5001/v1/embeddings \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer my-secret-key" \
    -d '{
      "input": "Hello, world!",
      "chunk": false,
      "save_to_qdrant": false
    }'
  ```

- **Example Response:**

  ```json
  {
    "object": "list",
    "data": [
      {
        "object": "embedding",
        "embedding": [0.123, -0.456, ...],
        "index": 0
      }
    ]
  }
  ```

---

### 3. Search Embeddings

- **Endpoint:** `POST /v1/search`
- **Authentication:**
  Include header `Authorization: Bearer <API_KEY>`
- **Request Body Options:**

  - **`query`**: A string query to search for relevant embeddings.
  - **`collection`**: (Optional) Qdrant collection name.
  - **`top_k`**: (Optional) Number of top results to return (default is 3).

- **Example Request:**

  ```bash
  curl -X POST http://localhost:5001/v1/search \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer my-secret-key" \
    -d '{
      "query": "Find relevant text",
      "top_k": 3
    }'
  ```

- **Example Response:**

  ```json
  {
    "object": "list",
    "data": [
      {
        "object": "search_result",
        "score": 0.98,
        "text": "Relevant text snippet..."
      }
    ]
  }
  ```

---

## Running with Docker

### 1. Run Locally with Docker (Recommended)

```sh
docker run -d -p 5001:5001 \
  --name embedding_service \
  -v .:/app \
  ghcr.io/ajianaz/embedding-service:latest
```

This command will:

- Run the container in detached mode (`-d`)
- Expose port `5001`
- Mount the application folder as a volume (for easy updates)

Test the service with:

```sh
curl -X POST http://localhost:5001/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-key" \
  -d '{"input": "Hello, world!"}'
```

---

### 2. Pull Image from GHCR

If you prefer not to build the image manually, pull the pre-built image:

```sh
docker pull ghcr.io/ajianaz/embedding-service:latest
```

Then run it using the Docker run command above.

---

### 3. Build & Run Locally (Optional)

To build your own image:

```sh
docker build -t embedding-service .
docker run -d -p 5001:5001 -v .:/app embedding-service
```

---

## Deploy with Docker Compose & Traefik

Create a `docker-compose.yml` file similar to the following example:

```yaml
version: '3.8'

networks:
  traefik-network:
    external: true

services:
  embedding:
    image: ghcr.io/ajianaz/embedding-service:latest
    container_name: embedding_service
    environment:
      - WORKERS=2
    command:
      ['sh', '-c', 'gunicorn -w ${WORKERS:-2} -b 0.0.0.0:5001 app.embedder:app']
    networks:
      - traefik-network
    volumes:
      - .:/app # Mount the local app folder into the container
    labels:
      - 'traefik.enable=true'
      - 'traefik.http.routers.embedding.rule=Host(`YOUR-DOMAIN`)'
      - 'traefik.http.routers.embedding.entrypoints=websecure'
      - 'traefik.http.routers.embedding.tls.certresolver=myresolver'
      - 'traefik.http.services.embedding.loadbalancer.server.port=5001'
```

Then run:

```sh
docker-compose up -d
```

---

## Environment Variables

The service uses the following environment variables (with defaults):

- **`API_KEY`**: Secret key for bearer authentication (default: `"my-secret-key"`).
- **`DEFAULT_CHUNK`**: Enable text chunking by default (`true` or `false`, default is `false`).
- **`DEFAULT_SAVE_QDRANT`**: Enable saving embeddings to Qdrant by default (`true` or `false`, default is `false`).
- **`OPTIMIZE_TEXT`**: Enable text optimization (normalization, stopword removal, and lemmatization) by default (`true` or `false`, default is `false`).

---

## License

MIT License. Feel free to use and modify. 😊
