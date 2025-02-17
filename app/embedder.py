from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from app.utils import test_qdrant_connection, chunk_text, save_to_qdrant, search_in_qdrant, logger, DEFAULT_COLLECTION

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_SIZE = model.get_sentence_embedding_dimension()

@app.route("/v1/embeddings", methods=["POST"])
def embed():
    data = request.json
    input_text = data.get("input", "")
    collection_name = data.get("collection", DEFAULT_COLLECTION)
    payload = data.get("metadata", {})
    chunk_size = data.get("chunk_size", 256)
    overlap = data.get("overlap", 50)

    if not input_text:
        return jsonify({"error": "Input text is required"}), 400

    chunks = chunk_text(input_text, chunk_size, overlap)
    embeddings = [model.encode(chunk).tolist() for chunk in chunks]

    for i, chunk in enumerate(chunks):
        save_to_qdrant(embeddings[i], chunk, collection_name, payload)

    return jsonify({"object": "embedding", "data": embeddings})

@app.route("/v1/search", methods=["POST"])
def search():
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