import os
from functools import wraps
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from app.utils import (test_qdrant_connection, chunk_text, save_to_qdrant, 
                       search_in_qdrant, logger, DEFAULT_COLLECTION, ensure_collection_exists)
from app.text_utils import optimize_text
from qdrant_client.models import Distance

app = Flask(__name__)

# Global cache untuk instance model
models_cache = {}

# Model default, dibaca dari environment variable MODEL_NAME
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-large-instruct")

def get_model(model_name=DEFAULT_MODEL_NAME):
    """
    Mengembalikan instance model dari cache jika sudah ada, atau memuat model baru dan menyimpannya.
    """
    if model_name in models_cache:
        return models_cache[model_name]
    try:
        model = SentenceTransformer(model_name)
        models_cache[model_name] = model
        return model
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        raise e

# Inisialisasi model default dan dapatkan vector size
default_model = get_model()
VECTOR_SIZE = default_model.get_sentence_embedding_dimension()

# Environment Variables untuk autentikasi dan konfigurasi lain
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
    """
    Endpoint untuk menampilkan daftar model yang tersedia.
    Jika environment variable MODELS_LIST disediakan (misalnya: "model1, model2, model3"),
    maka akan menampilkan daftar tersebut. Jika tidak, maka akan menampilkan model yang sudah
    dimuat dalam cache. Jika cache kosong, maka akan menggunakan default model.
    """
    models_from_env = os.getenv("MODELS_LIST")
    if models_from_env:
        models = [m.strip() for m in models_from_env.split(",") if m.strip()]
    else:
        models = list(models_cache.keys())
        if not models:
            models = [DEFAULT_MODEL_NAME]
    
    response_data = [{
        "id": model_name,
        "object": "model",
        "owned_by": "ajianaz-dev"
    } for model_name in models]
    
    return jsonify({
        "object": "list",
        "data": response_data
    })

@app.route("/v1/optimize", methods=["POST"])
@authenticate
def optimize_route():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text is required"}), 400

    # Dapatkan parameter replace_symbols, default False (bisa berupa boolean atau string)
    replace_symbols = data.get("replace_symbols", False)
    # Jika parameter berupa string, konversi ke boolean
    if isinstance(replace_symbols, str):
        replace_symbols = replace_symbols.lower() == "true"

    optimized = optimize_text(text, replace_symbols)
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

    # Pilih model berdasarkan parameter request, jika tidak ada gunakan default
    model_name = data.get("model", DEFAULT_MODEL_NAME)
    try:
        current_model = get_model(model_name)
    except Exception as e:
        return jsonify({"error": f"Failed to load model '{model_name}'", "details": str(e)}), 500

    # Validasi input (harus string atau list of strings)
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
                chunk = optimize_text(chunk)

            try:
                embedding = current_model.encode(chunk).tolist()
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

@app.route("/v1/collection", methods=["POST"])
@authenticate
def create_collection():
    """
    Endpoint untuk membuat collection baru di Qdrant.
    Request JSON harus mengandung:
      - collection_name (str): Nama collection.
      - vector_size (int): Ukuran vektor. Default: 384.
      - distance (str, opsional): Metode distance. Default: "COSINE".
    """
    data = request.json
    collection_name = data.get("collection_name")
    vector_size = data.get("vector_size", 384)
    distance_str = data.get("distance", "COSINE")
    
    if collection_name is None:
        return jsonify({"error": "collection_name harus disediakan"}), 400

    try:
        vector_size = int(vector_size)
    except Exception as e:
        return jsonify({"error": "vector_size harus berupa integer", "details": str(e)}), 400

    # Konversi string distance menjadi enum Distance (default COSINE)
    distance_value = getattr(Distance, distance_str.upper(), Distance.COSINE)

    try:
        ensure_collection_exists(collection_name, vector_size, distance_value)
    except Exception as e:
        return jsonify({"error": "Gagal membuat collection", "details": str(e)}), 500

    return jsonify({
        "object": "status",
        "message": f"Collection '{collection_name}' sudah tersedia atau berhasil dibuat."
    })

@app.route("/v1/search", methods=["POST"])
@authenticate
def search():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Pilih model berdasarkan parameter request, jika tidak gunakan default
    model_name = data.get("model", DEFAULT_MODEL_NAME)
    try:
        current_model = get_model(model_name)
    except Exception as e:
        return jsonify({"error": f"Failed to load model '{model_name}'", "details": str(e)}), 500

    collection_name = data.get("collection", DEFAULT_COLLECTION)
    top_k = int(data.get("top_k", 3))
    
    score_threshold = data.get("score_threshold", None)
    if score_threshold is not None:
        try:
            score_threshold = float(score_threshold)
        except ValueError:
            return jsonify({"error": "Invalid score_threshold value, must be numeric."}), 400

    # Kumpulkan parameter tambahan secara dinamis (kecuali beberapa parameter yang sudah diketahui)
    dynamic_params = { key: value for key, value in data.items() 
                       if key not in ["query", "collection", "top_k", "score_threshold", "model"] }

    try:
        query_embedding = current_model.encode([query])[0].tolist()
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}")
        return jsonify({"error": "Failed to generate query embedding", "details": str(e)}), 500

    results = search_in_qdrant(query_embedding, collection_name, top_k, **dynamic_params)
    formatted_results = results.get("data", [])
    if score_threshold is not None:
        formatted_results = [res for res in formatted_results if res.get("score", 0) >= score_threshold]

    return jsonify({"object": "list", "data": formatted_results})

if test_qdrant_connection():
    logger.info("Qdrant is ready to use")
else:
    logger.warning("Qdrant is not available, some features may not work")

# Uncomment baris di bawah untuk menjalankan server secara langsung (local testing)
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001, debug=True)