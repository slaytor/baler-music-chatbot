import os
from pathlib import Path
from dotenv import load_dotenv

# --- PROJECT ROOT ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- LOAD ENVIRONMENT VARIABLES ---
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

# --- DATABASE PROVIDER ---
DB_PROVIDER = os.getenv("DB_PROVIDER", "CLOUD")

# --- API CONFIGURATION ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "OLLAMA")
GEMINI_MODEL = "gemini-2.0-flash" # Set to specific version
OLLAMA_MODEL = "mistral" # Switched to Mistral for better instruction following
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

# --- AUTHENTICATION ---
CREDENTIALS_FILE_NAME = "gcloud-credentials.json"
CREDENTIALS_PATH = PROJECT_ROOT / CREDENTIALS_FILE_NAME
TOKEN_LIFESPAN_SECONDS = 45 * 60  # 45 minutes for Gemini

# --- KNOWLEDGE BASE CONFIG ---
# Local ChromaDB configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# Chroma Cloud configuration (loaded from environment variables)
CHROMA_CLOUD_API_KEY = os.getenv("CHROMA_CLOUD_API_KEY")
CHROMA_CLOUD_TENANT = os.getenv("CHROMA_CLOUD_TENANT")
CHROMA_CLOUD_DATABASE = os.getenv("CHROMA_CLOUD_DATABASE")

COLLECTION_NAME = "pitchfork_reviews"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- SCRAPER & DATA CONFIG ---
RAW_DATA_FILE = PROJECT_ROOT / "reviews.jsonl"
S3_BUCKET_NAME = "baler-music-chatbot"
S3_DAILY_PREFIX = "daily_scrapes/"
PROCESSED_FILES_LOG = PROJECT_ROOT / ".processed_s3_files.log"

# --- PIPELINE BATCHING ---
KB_BATCH_SIZE = 5
INTER_BATCH_DELAY_SECONDS = 1
