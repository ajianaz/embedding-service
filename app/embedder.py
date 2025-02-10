import os
import uuid
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Konfigurasi Qdrant
QDRANT_ENABLE = os.getenv("QDRANT_ENABLE", "false").lower() == "true"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# Inisialisasi Qdrant Client jika diaktifkan
qdrant_client = None
if QDRANT_ENABLE:
    qdrant_client = QdrantClient(
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}" if ":" in QDRANT_HOST else f"https://{QDRANT_HOST}",
        api_key=QDRANT_API_KEY
    )

# Inisialisasi Flask dan model embedding
app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_SIZE = 384  # Sesuaikan dengan dimensi model embedding


def ensure_collection_exists(collection_name):
    """Memastikan koleksi ada di Qdrant, jika tidak ada maka buat baru."""
    if not QDRANT_ENABLE or not qdrant_client:
        return

    existing_collections = qdrant_client.get_collections().collections
    collection_names = [col.name for col in existing_collections]

    if collection_name not in collection_names:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )


def chunk_text(text, chunk_size=256, overlap=50):
    """ Membagi teks panjang menjadi beberapa bagian dengan overlap """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks


def save_to_qdrant(embedding, text, payload, collection_name="qdrant_messages"):
    """ Menyimpan embedding ke Qdrant """
    if not QDRANT_ENABLE or not qdrant_client:
        return
    
    ensure_collection_exists(collection_name)  # Pastikan koleksi ada sebelum menyimpan

    point_id = str(uuid.uuid4())  # Generate UUID v4 untuk ID unik
    
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={"text": text, **payload}  # Menyimpan teks + custom payload
            )
        ]
    )


def search_in_qdrant(text, top_k=3, collection_name="qdrant_messages"):
    """ Mencari embedding di Qdrant dengan chunking jika teks panjang """
    if not QDRANT_ENABLE or not qdrant_client:
        return []
    
    ensure_collection_exists(collection_name)  # Pastikan koleksi ada sebelum pencarian

    chunks = chunk_text(text)
    results = []

    for chunk in chunks:
        embedding = model.encode([chunk])[0].tolist()
        
        qdrant_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=top_k
        )

        for result in qdrant_results:
            results.append({
                "text": result.payload.get("text"),
                **{k: v for k, v in result.payload.items() if k != "text"},
                "score": result.score
            })
    
    # Urutkan berdasarkan skor tertinggi
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]  # Ambil top-k hasil terbaik


@app.route("/embed", methods=["POST"])
def embed():
    """ API endpoint untuk embedding teks dan menyimpan ke Qdrant jika diaktifkan """
    data = request.json
    text = data.get("text", "")
    payload = data.get("payload", {})  # Bisa berisi user_id, chat_id, dll.
    collection_name = data.get("collection_name", "qdrant_messages")

    if not text:
        return jsonify({"error": "Text is required"}), 400

    # Jika teks panjang, gunakan chunking
    chunks = chunk_text(text)
    embeddings = [model.encode([chunk])[0].tolist() for chunk in chunks]

    if QDRANT_ENABLE:
        for chunk, embedding in zip(chunks, embeddings):
            save_to_qdrant(embedding, chunk, payload, collection_name)

    return jsonify({"embeddings": embeddings})


@app.route("/search", methods=["POST"])
def search():
    """ API endpoint untuk mencari teks di Qdrant """
    data = request.json
    text = data.get("text", "")
    top_k = data.get("top_k", 3)
    collection_name = data.get("collection_name", "qdrant_messages")

    if not text:
        return jsonify({"error": "Text is required"}), 400

    results = search_in_qdrant(text, top_k, collection_name)
    return jsonify({"results": results})


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001, debug=True)