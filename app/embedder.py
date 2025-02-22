import os
from functools import wraps
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from app.utils import test_qdrant_connection, chunk_text, save_to_qdrant, search_in_qdrant, logger, DEFAULT_COLLECTION
from app.text_utils import optimize_text, remove_stopwords, stem_text, lemmatize_text

app = Flask(__name__)

MODEL_NAME = "paraphrase-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
VECTOR_SIZE = model.get_sentence_embedding_dimension()

# Environment Variables
API_KEY = os.getenv("API_KEY", "my-secret-key")
DEFAULT_CHUNK = os.getenv("DEFAULT_CHUNK", "false")
DEFAULT_SAVE_QDRANT = os.getenv("DEFAULT_SAVE_QDRANT", "false")
DEFAULT_OPTIMIZE_TEXT = os.getenv("OPTIMIZE_TEXT", "false")

# Helper: konversi string ke boolean
def str_to_bool(value):
    return str(value).lower() == "true"

DEFAULT_CHUNK = str_to_bool(DEFAULT_CHUNK)
DEFAULT_SAVE_QDRANT = str_to_bool(DEFAULT_SAVE_QDRANT)
DEFAULT_OPTIMIZE_TEXT = str_to_bool(DEFAULT_OPTIMIZE_TEXT)

# Decorator untuk autentikasi
def authenticate(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if auth_header != f"Bearer {API_KEY}":
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

# Helper untuk memproses chunking teks
def process_text(text, chunk_enabled, chunk_size, overlap):
    if chunk_enabled:
        try:
            return chunk_text(text, chunk_size, overlap)
        except Exception as e:
            logger.error(f"Error during chunking text: {e}")
            raise e
    return [text]

@app.route("/v1/models", methods=["GET"])
@authenticate
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
@authenticate
def optimize_route():
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
@authenticate
def chunk_route():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text is required"}), 400
    chunk_size = data.get("chunk_size", 256)
    overlap = data.get("overlap", 50)
    try:
        chunks = chunk_text(text, chunk_size, overlap)
    except Exception as e:
        logger.error(f"Error during chunking text: {e}")
        return jsonify({"error": "Failed to chunk text", "details": str(e)}), 500
    return jsonify({
        "object": "chunks",
        "data": chunks
    })

@app.route("/v1/embeddings", methods=["POST"])
@authenticate
def embed():
    data = request.json

    # Validasi input (harus string atau list)
    input_text = data.get("input", "")
    if not isinstance(input_text, (str, list)):
        return jsonify({"error": "Input text must be a string or list of strings"}), 400
    texts = input_text if isinstance(input_text, list) else [input_text]

    collection_name = data.get("collection", DEFAULT_COLLECTION)
    payload = data.get("metadata", {})
    chunk_size = data.get("chunk_size", 256)
    overlap = data.get("overlap", 50)

    # Konversi opsi dari request ke boolean
    chunk_enabled = str_to_bool(data.get("chunk", DEFAULT_CHUNK))
    save_enabled = str_to_bool(data.get("save_to_qdrant", DEFAULT_SAVE_QDRANT))
    optimize_flag = str_to_bool(data.get("optimize_text", DEFAULT_OPTIMIZE_TEXT))
    optimize_next_step = str_to_bool(data.get("optimize_next_step", "false"))

    embeddings_results = []
    processed_count = 0
    index = 0

    for text in texts:
        try:
            chunks = process_text(text, chunk_enabled, chunk_size, overlap)
        except Exception as e:
            return jsonify({"error": "Failed to process text", "details": str(e)}), 500

        for chunk in chunks:
            if optimize_flag:
                # Langkah pertama: normalisasi (lowercase, hapus simbol)
                chunk = optimize_text(chunk)
                # Jika optimize_next_step aktif, lakukan penghapusan stopwords dan lemmatization
                if optimize_next_step:
                    chunk = remove_stopwords(chunk)
                    chunk = lemmatize_text(chunk)

            try:
                embedding = model.encode(chunk).tolist()
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                return jsonify({"error": "Failed to generate embeddings", "details": str(e)}), 500

            if save_enabled:
                try:
                    save_to_qdrant(embedding, chunk, collection_name, payload)
                except Exception as e:
                    logger.error(f"Failed to save embeddings to Qdrant: {e}")
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
    return jsonify({
        "object": "list",
        "data": embeddings_results
    })

@app.route("/v1/search", methods=["POST"])
@authenticate
def search():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    collection_name = data.get("collection", DEFAULT_COLLECTION)
    top_k = int(data.get("top_k", 3))
    
    # Ambil parameter score_threshold dari payload (opsional)
    score_threshold = data.get("score_threshold", None)
    if score_threshold is not None:
        try:
            score_threshold = float(score_threshold)
        except ValueError:
            return jsonify({"error": "Invalid score_threshold value, must be numeric."}), 400

    # Kumpulkan parameter tambahan secara dinamis dari payload
    dynamic_params = {}
    for key, value in data.items():
        if key not in ["query", "collection", "top_k", "score_threshold"]:
            dynamic_params[key] = value

    try:
        query_embedding = model.encode([query])[0].tolist()
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}")
        return jsonify({"error": "Failed to generate query embedding", "details": str(e)}), 500

    # Panggil fungsi pencarian. Misalnya, jika fungsi search_in_qdrant mendukung parameter tambahan.
    results = search_in_qdrant(query_embedding, collection_name, top_k, **dynamic_params)
    formatted_results = results.get("data", [])

    # Jika score_threshold diberikan, filter hasil dengan score yang memenuhi threshold
    if score_threshold is not None:
        formatted_results = [res for res in formatted_results if res.get("score", 0) >= score_threshold]

    return jsonify({"object": "list", "data": formatted_results})


if test_qdrant_connection():
    logger.info("Qdrant is ready to use")
else:
    logger.warning("Qdrant is not available, some features may not work")

# Uncomment baris berikut untuk menjalankan server secara langsung (local testing)
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001, debug=True)