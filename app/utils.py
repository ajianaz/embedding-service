import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import uuid

# Load environment variables
QDRANT_ENABLE = os.getenv("QDRANT_ENABLE", "False").lower() == "true"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default vector size (should match the embedding model output size)
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 produces 384-dim vectors

def get_qdrant_client():
    """Initialize Qdrant connection and create collection if necessary."""
    if not QDRANT_ENABLE:
        logger.info("❌ Qdrant is disabled in .env")
        return None

    try:
        qdrant_kwargs = {"host": QDRANT_HOST}
        if QDRANT_API_KEY:
            qdrant_kwargs["api_key"] = QDRANT_API_KEY
        else:
            qdrant_kwargs["port"] = QDRANT_PORT

        client = QdrantClient(**qdrant_kwargs)
        client.get_collections()  # Check connection
        logger.info("✅ Qdrant connection established")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant: {e}")
        return None

def ensure_collection_exists(client, collection_name):
    """Ensure collection exists in Qdrant, create if necessary."""
    try:
        collections = client.get_collections().collections
        existing_collections = {c.name for c in collections}

        if collection_name not in existing_collections:
            client.create_collection(
                collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            logger.info(f"✅ Collection '{collection_name}' created")

    except Exception as e:
        logger.error(f"❌ Error ensuring collection '{collection_name}': {e}")

def save_to_qdrant(client, collection_name, doc_id, embedding, text, payload):
    """Save embedding and metadata to Qdrant."""
    try:
        ensure_collection_exists(client, collection_name)
        payload["text"] = text  # Ensure text is always included
        client.upsert(
            collection_name=collection_name,
            points=[{
                "id": doc_id,
                "vector": embedding,
                "payload": payload
            }]
        )
        logger.info(f"✅ Saved document {doc_id} to '{collection_name}'")
    except Exception as e:
        logger.error(f"❌ Failed to save document {doc_id} to Qdrant: {e}")

def search_in_qdrant(client, collection_name, query_embedding, top_k=3):
    """Search similar embeddings in Qdrant."""
    try:
        ensure_collection_exists(client, collection_name)
        results = client.search(collection_name, query_vector=query_embedding, limit=top_k)
        return [r.payload.get("text", "") for r in results]
    except Exception as e:
        logger.error(f"❌ Qdrant search failed: {e}")
        return []

def chunk_text(text, chunk_size=256, overlap=50):
    """Chunk text into overlapping parts."""
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks