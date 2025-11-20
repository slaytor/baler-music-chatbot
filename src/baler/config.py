import os
from pathlib import Path

# --- PROJECT ROOT ---
# (2 levels up from this file: src/baler -> src -> baler-music-chatbot)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# --- ENVIRONMENT & AUTH ---
# Load .env file from the project root
ENV_PATH = PROJECT_ROOT / ".env"
CREDENTIALS_FILE_NAME = "gcloud-credentials.json"
CREDENTIALS_PATH = PROJECT_ROOT / CREDENTIALS_FILE_NAME


# --- API CONFIGURATION ---
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
TOKEN_LIFESPAN_SECONDS = 45 * 60  # 45 minutes


# --- SCRAPER & DATA CONFIG ---
RAW_DATA_FILE = PROJECT_ROOT / "reviews.jsonl"
S3_BUCKET_NAME = "baler-music-chatbot"
S3_DAILY_PREFIX = "daily_scrapes/"
PROCESSED_FILES_LOG = PROJECT_ROOT / ".processed_s3_files.log"


# --- KNOWLEDGE BASE CONFIG ---
# Connect to ChromaDB over the network
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = "pitchfork_reviews"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# --- PIPELINE BATCHING ---
# Number of reviews to process at a time (for create_knowledge_base)
KB_BATCH_SIZE = 5
# Number of concurrent API calls (for create_knowledge_base)
CONCURRENT_REQUESTS = 10
# Delay between concurrent batches
INTER_BATCH_DELAY_SECONDS = 1
