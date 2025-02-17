from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from app.utils import test_qdrant_connection, chunk_text, save_to_qdrant, search_in_qdrant, logger, DEFAULT_COLLECTION
import os

app = Flask(__name__)
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
VECTOR_SIZE = model.get_sentence_embedding_dimension()
API_KEY = os.getenv("API_KEY", "$2a$10$8E2cnsRvDhLnjxvQK7AJlujKNu324RkimgYJrZ8CeBJ7z1vMH0sJu")
DEFAULT_CHUNK = os.getenv("DEFAULT_CHUNK", "false").lower() == "true"

@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "ajianaz-dev"
            }
        ]
    })

@app.route("/v1/embeddings", methods=["POST"])
def embed():
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    input_text = data.get("input", "")
    collection_name = data.get("collection", DEFAULT_COLLECTION)
    payload = data.get("metadata", {})
    chunk_size = data.get("chunk_size", 256)
    overlap = data.get("overlap", 50)

    chunk_enabled = data.get("chunk", DEFAULT_CHUNK)
    if not isinstance(input_text, str):
        return jsonify({"error": "Input text must be a string"}), 400

    if not input_text:
        return jsonify({"error": "Input text is required"}), 400

    # Jika chunking diaktifkan, pecah teks; jika tidak, proses teks utuh
    if chunk_enabled:
        try:
            chunks = chunk_text(input_text, chunk_size, overlap)
        except Exception as e:
            return jsonify({"error": "Failed to chunk text", "details": str(e)}), 500
    else:
        chunks = [input_text]

    try:
        # Proses embedding
        embeddings = [model.encode(chunk).tolist() for chunk in chunks]
    except Exception as e:
        return jsonify({"error": "Failed to generate embeddings", "details": str(e)}), 500

    try:
        # Simpan ke Qdrant
        for i, chunk in enumerate(chunks):
            save_to_qdrant(embeddings[i], chunk, collection_name, payload)
    except Exception as e:
        return jsonify({"error": "Failed to save embeddings to Qdrant", "details": str(e)}), 500

    return jsonify({
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": emb, "index": i} for i, emb in enumerate(embeddings)
        ]
    })

@app.route("/v1/search", methods=["POST"])
def search():
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    query = data.get("query", "")
    collection_name = data.get("collection", DEFAULT_COLLECTION)
    top_k = int(data.get("top_k", 3))

    if not query:
        return jsonify({"error": "Query is required"}), 400

    query_embedding = model.encode([query])[0].tolist()
    results = search_in_qdrant(query_embedding, collection_name, top_k)

    formatted_results = [
        {"object": "search_result", "score": res["score"], "text": res["text"]}
        for res in results
    ]

    return jsonify({"object": "list", "data": formatted_results})

if test_qdrant_connection():
    logger.info("Qdrant is ready to use")
else:
    logger.warning("Qdrant is not available, some features may not work")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001, debug=True)