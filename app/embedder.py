from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from utils import chunk_text, save_to_qdrant, search_in_qdrant, logger, DEFAULT_COLLECTION

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_SIZE = model.get_sentence_embedding_dimension()  # Pastikan ini benar

@app.route("/embed", methods=["POST"])
def embed():
    data = request.json
    text = data.get("text", "")
    collection_name = data.get("collection_name", DEFAULT_COLLECTION)
    payload = data.get("payload", {})

    if not text:
        return jsonify({"error": "Text is required"}), 400

    # Proses chunking jika teks panjang
    chunks = chunk_text(text)
    embeddings = [model.encode(chunk).tolist() for chunk in chunks]

    # Simpan ke Qdrant jika diaktifkan
    for i, chunk in enumerate(chunks):
        save_to_qdrant(embeddings[i], chunk, collection_name, payload)

    return jsonify({"embeddings": embeddings})

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "")
    collection_name = data.get("collection_name", DEFAULT_COLLECTION)
    top_k = int(data.get("top_k", 3))

    if not query:
        return jsonify({"error": "Query is required"}), 400

    query_embedding = model.encode([query])[0].tolist()
    results = search_in_qdrant(query_embedding, collection_name, top_k)

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)