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

# Load environment variables
QDRANT_ENABLE = os.getenv("QDRANT_ENABLE", "False").lower() == "true"
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)  # API Key if used
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "qdrant_messages")
PREFER_GRPC = os.getenv("PREFER_GRPC", "False").lower() == "true"

# Initialize Qdrant client if enabled
qdrant_client = None
if QDRANT_ENABLE:
    try:
        # Tentukan protokol berdasarkan apakah API key digunakan atau tidak
        # if QDRANT_API_KEY:
        #     if not QDRANT_HOST.startswith("https://"):
        #         raise ValueError("❌ QDRANT_API_KEY detected, but QDRANT_HOST must use HTTPS")
        #     scheme = "https"
        # else:
        #     scheme = "http"

        # Jika QDRANT_HOST sudah punya skema (http/https), gunakan langsung
        if QDRANT_HOST.startswith("http://") or QDRANT_HOST.startswith("https://"):
            qdrant_url = QDRANT_HOST
        else:
            qdrant_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

        # Inisialisasi Qdrant Client
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=QDRANT_API_KEY or None,
            prefer_grpc=PREFER_GRPC
        )

        # Tes koneksi
        qdrant_client.get_collections()
        logger.info(f"✅ Connected to Qdrant at {qdrant_url}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant: {e}")
        qdrant_client = None

def test_qdrant_connection():
    if qdrant_client is None:
        logger.error("❌ Qdrant client is not initialized")
        return False

    try:
        qdrant_client.get_collections()
        logger.info("✅ Qdrant connection test successful")
        return True
    except Exception as e:
        logger.error(f"❌ Qdrant connection test failed: {e}")
        return False

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
def chunk_text(text, chunk_size=256, overlap=50, separator=" "):
    if not isinstance(text, str):
        raise ValueError("Expected text to be a string, but got {}".format(type(text)))

    words = text.split(separator)
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(separator.join(words[i:i + chunk_size]))
    
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
        query_vector=embedding,
        limit=top_k
    )
    return [r.payload["text"] for r in results]
