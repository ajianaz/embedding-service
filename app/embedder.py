import os
import uuid
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import qdrant_client

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")


QDRANT_ENABLE = os.getenv("QDRANT_ENABLE", "false").lower() == "true"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# Inisialisasi Qdrant hanya jika diaktifkan
qdrant = None
if QDRANT_ENABLE:
    if "http" in QDRANT_HOST:  # Jika sudah berupa URL, gunakan langsung
        qdrant = qdrant_client.QdrantClient(url=QDRANT_HOST)
    else:  # Jika pakai IP atau domain tanpa URL, gunakan port
        qdrant = qdrant_client.QdrantClient(host=QDRANT_HOST, port=int(QDRANT_PORT))


# Fungsi untuk memecah teks menjadi chunk dengan overlap
def chunk_text(text, chunk_size=256, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


# Fungsi menyimpan embedding ke Qdrant (jika diaktifkan)
def save_to_qdrant(embedding, text, payload={}):
    if not QDRANT_ENABLE:
        return

    # Pastikan payload tetap ada default text seperti sebelumnya
    payload = {**{"type": "message", "text": text}, **payload}

    qdrant.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[{
            "id": str(uuid.uuid4()),  # UUID v4 sebagai ID unik
            "vector": embedding,
            "payload": payload
        }]
    )


@app.route("/embed", methods=["POST"])
def embed():
    data = request.json
    text = data.get("text", "")
    payload = data.get("payload", {})  # Bisa berisi user_id, chat_id, dll.

    if not text:
        return jsonify({"error": "Text is required"}), 400

    chunks = chunk_text(text)
    embeddings = [model.encode(chunk).tolist() for chunk in chunks]

    if QDRANT_ENABLE:
        for chunk, embedding in zip(chunks, embeddings):
            save_to_qdrant(embedding, chunk, payload)

    return jsonify({"embeddings": embeddings})


@app.route("/search", methods=["POST"])
def search():
    if not QDRANT_ENABLE:
        return jsonify({"error": "Qdrant is not enabled"}), 400

    data = request.json
    query_text = data.get("text", "")
    top_k = data.get("top_k", 3)

    if not query_text:
        return jsonify({"error": "Query text is required"}), 400

    query_embedding = model.encode([query_text])[0].tolist()
    results = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_embedding,
        limit=top_k
    )

    return jsonify({"results": [r.payload for r in results]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)