from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from app.utils import test_qdrant_connection, chunk_text, save_to_qdrant, search_in_qdrant, logger, DEFAULT_COLLECTION
from app.text_utils import optimize_text, remove_stopwords, stem_text, lemmatize_text
import os

app = Flask(__name__)

MODEL_NAME = "paraphrase-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
VECTOR_SIZE = model.get_sentence_embedding_dimension()

API_KEY = os.getenv("API_KEY", "my-secret-apikey")
DEFAULT_CHUNK = os.getenv("DEFAULT_CHUNK", "false").lower() == "true"
DEFAULT_SAVE_QDRANT = os.getenv("DEFAULT_SAVE_QDRANT", "false").lower() == "true"
DEFAULT_OPTIMIZE_TEXT = os.getenv("OPTIMIZE_TEXT", "false").lower() == "true"

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

@app.route("/v1/optimize", methods=["POST"])
def optimize_route():
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text is required"}), 400

    optimized = optimize_text(text)
    return jsonify({
        "object": "optimized_text",
        "text": optimized
    })

@app.route("/v1/chunk", methods=["POST"])
def chunk_route():
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text is required"}), 400

    chunk_size = data.get("chunk_size", 256)
    overlap = data.get("overlap", 50)

    try:
        chunks = chunk_text(text, chunk_size, overlap)
    except Exception as e:
        return jsonify({"error": "Failed to chunk text", "details": str(e)}), 500

    return jsonify({
        "object": "chunks",
        "data": chunks
    })

@app.route("/v1/embeddings", methods=["POST"])
def embed():
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json

    # Mendukung input berupa string atau list
    input_text = data.get("input", "")
    if isinstance(input_text, list):
        texts = input_text
    elif isinstance(input_text, str):
        texts = [input_text]
    else:
        return jsonify({"error": "Input text must be a string or list of strings"}), 400

    collection_name = data.get("collection", DEFAULT_COLLECTION)
    payload = data.get("metadata", {})
    chunk_size = data.get("chunk_size", 256)
    overlap = data.get("overlap", 50)

    # Opsi chunking, default dari env
    chunk_enabled = data.get("chunk", DEFAULT_CHUNK)
    if isinstance(chunk_enabled, str):
        chunk_enabled = chunk_enabled.lower() == "true"

    # Opsi penyimpanan ke Qdrant, default dari env
    save_enabled = data.get("save_to_qdrant", DEFAULT_SAVE_QDRANT)
    if isinstance(save_enabled, str):
        save_enabled = save_enabled.lower() == "true"

    # Opsi optimize text, default dari env
    optimize_flag = data.get("optimize_text", DEFAULT_OPTIMIZE_TEXT)
    if isinstance(optimize_flag, str):
        optimize_flag = optimize_flag.lower() == "true"

    embeddings_results = []
    index = 0
    processed_count = 0

    for text in texts:
        # Jika chunking diaktifkan, bagi input menjadi chunk
        if chunk_enabled:
            try:
                chunks = chunk_text(text, chunk_size, overlap)
            except Exception as e:
                return jsonify({"error": "Failed to chunk text", "details": str(e)}), 500
        else:
            chunks = [text]

        for chunk in chunks:
            # Lakukan optimasi teks jika diaktifkan
            if optimize_flag:
                # Normalize text: lowercase dan hapus tanda/simbol khusus
                chunk = optimize_text(chunk)
                # Hapus stopwords
                chunk = remove_stopwords(chunk)
                # Lakukan lemmatization
                chunk = lemmatize_text(chunk)

            try:
                embedding = model.encode(chunk).tolist()
            except Exception as e:
                return jsonify({"error": "Failed to generate embeddings", "details": str(e)}), 500

            if save_enabled:
                try:
                    save_to_qdrant(embedding, chunk, collection_name, payload)
                except Exception as e:
                    return jsonify({"error": "Failed to save embeddings to Qdrant", "details": str(e)}), 500
                processed_count += 1
            else:
                embeddings_results.append({
                    "object": "embedding",
                    "embedding": embedding,
                    "index": index
                })
                index += 1

    if save_enabled:
        return jsonify({
            "object": "status",
            "message": f"Processed and saved {processed_count} embedding(s) to Qdrant."
        })
    else:
        return jsonify({
            "object": "list",
            "data": embeddings_results
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
    formatted_results = results.get("data", [])

    return jsonify({"object": "list", "data": formatted_results})

if test_qdrant_connection():
    logger.info("Qdrant is ready to use")
else:
    logger.warning("Qdrant is not available, some features may not work")

# Uncomment baris berikut untuk menjalankan server secara langsung
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001, debug=True)