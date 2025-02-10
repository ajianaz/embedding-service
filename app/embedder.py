import os
import logging
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from utils import chunk_text, get_qdrant_client, save_to_qdrant, search_in_qdrant
import uuid

# Load environment variables
DEBUG_MODE = os.getenv("FLASK_DEBUG", "False").lower() == "true"
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "qdrant_messages")

# Configure logging
logging.basicConfig(level=logging.DEBUG if DEBUG_MODE else logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask and SentenceTransformer model
app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Qdrant client
qdrant_client = get_qdrant_client()

@app.route("/embed", methods=["POST"])
def embed():
    data = request.json
    text = data.get("text", "")
    payload = data.get("payload", {})  # Optional payload (e.g., user_id, chat_id, etc.)
    collection_name = data.get("collection_name", DEFAULT_COLLECTION)

    if not text:
        return jsonify({"error": "Text is required"}), 400

    # Chunk text if longer than 256 words
    chunks = chunk_text(text)

    # Get embeddings for each chunk
    embeddings = model.encode(chunks).tolist()

    # Save embeddings to Qdrant (if enabled)
    if qdrant_client:
        for chunk, embedding in zip(chunks, embeddings):
            doc_id = str(uuid.uuid4())  # Generate unique ID
            save_to_qdrant(qdrant_client, collection_name, doc_id, embedding, chunk, payload)

    return jsonify({"embeddings": embeddings})

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    text = data.get("text", "")
    collection_name = data.get("collection_name", DEFAULT_COLLECTION)
    top_k = int(data.get("top_k", 3))

    if not text:
        return jsonify({"error": "Text is required"}), 400

    # Encode text query
    query_embedding = model.encode([text])[0].tolist()

    # Search in Qdrant
    if qdrant_client:
        results = search_in_qdrant(qdrant_client, collection_name, query_embedding, top_k)
        return jsonify({"results": results})

    return jsonify({"error": "Qdrant is not enabled"}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001, debug=DEBUG_MODE)