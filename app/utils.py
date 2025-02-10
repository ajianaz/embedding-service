import os
import logging
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams

# Load ENV variables
load_dotenv()

FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"

# Logging setup
logging.basicConfig(level=logging.DEBUG if FLASK_DEBUG else logging.INFO)
logger = logging.getLogger(__name__)

# Qdrant Config
QDRANT_ENABLE = os.getenv("QDRANT_ENABLE", "False").lower() == "true"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)  # API Key jika digunakan
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "qdrant_messages")

# Inisialisasi Qdrant jika diaktifkan
qdrant_client = None
if QDRANT_ENABLE:
    try:
        qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
        )
        qdrant_client.get_collections()
        logger.info("✅ Qdrant connection established")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant: {e}")

# Fungsi untuk membuat collection jika belum ada
def ensure_collection_exists(collection_name, vector_size):
    if not QDRANT_ENABLE:
        return

    try:
        qdrant_client.get_collection(collection_name)
        logger.info(f"ℹ️ Collection '{collection_name}' already exists")
    except UnexpectedResponse:
        logger.info(f"⚡ Creating collection '{collection_name}'...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f"✅ Collection '{collection_name}' created successfully")

# Fungsi untuk chunking teks
def chunk_text(text, chunk_size=256, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# Fungsi untuk menyimpan embedding ke Qdrant
def save_to_qdrant(embedding, text, collection_name=DEFAULT_COLLECTION, payload={}):
    if not QDRANT_ENABLE:
        return

    ensure_collection_exists(collection_name, len(embedding))

    point_id = str(uuid.uuid4())
    payload_data = {"text": text, **payload}

    qdrant_client.upsert(
        collection_name=collection_name,
        points=[{
            "id": point_id,
            "vector": embedding,
            "payload": payload_data
        }]
    )
    logger.info(f"✅ Saved data to Qdrant (Collection: {collection_name}, ID: {point_id})")

# Fungsi untuk mencari di Qdrant
def search_in_qdrant(embedding, collection_name=DEFAULT_COLLECTION, top_k=3):
    if not QDRANT_ENABLE:
        return []

    results = qdrant_client.search(
        collection_name=collection_name,
        vector=embedding,
        limit=top_k
    )
    return [r.payload["text"] for r in results]