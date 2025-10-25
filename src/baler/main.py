import os
import httpx
import json
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Any
from dotenv import load_dotenv
from pathlib import Path
# --- THE FIX: Import CORSMiddleware ---
from fastapi.middleware.cors import CORSMiddleware


# Import the shared authentication function
from .auth_util import get_gcp_auth_token

# Load environment variables from .env file
load_dotenv()


# --- CONFIGURATION ---
MODEL_NAME = 'all-MiniLM-L6-v2'
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "pitchfork_reviews"


# --- INITIALIZATION ---
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_collection(name=COLLECTION_NAME)
print("ChromaDB connection successful.")

app = FastAPI(
    title="Baler Music Recommendation API",
    description="A chatbot for nuanced music recommendations from Pitchfork reviews.",
)

# --- THE FIX: Add CORS middleware to allow browser requests ---
# This allows your web browser (running on a different "origin") to make requests to your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# --- API DATA MODELS ---
class Query(BaseModel):
    text: str
    top_k: int = 5


class Recommendation(BaseModel):
    response: str
    sources: list[dict]


# --- CORE RAG LOGIC ---

def search_reviews(query_text: str, top_k: int) -> List[Any]:
    """
    Performs a direct semantic search against the tag-enriched knowledge base.
    """
    query_embedding = model.encode([query_text]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    # The metadata contains the original text chunk and all other review info
    return [metadata for metadata in results['metadatas'][0]]


async def generate_response(query_text: str, context_chunks: list[dict]) -> dict:
    """
    Generates a response using the Google Gemini API with a specific persona.
    """
    auth_token = get_gcp_auth_token()
    if not auth_token:
        return {"response": "Could not authenticate with Google Cloud.", "sources": []}

    context_str = "\n\n".join([f"From a review of '{chunk['album_title']}' by {chunk['artist']}:\n...{chunk['text_chunk']}..." for chunk in context_chunks])

    system_prompt = (
        "You are Baler, an AI music critic in the style of a Pitchfork reviewer. You are "
        "knowledgeable, a little bit pretentious, and have a distinctive voice. Your "
        "recommendations must be based ONLY on the provided review excerpts. Justify your "
        "suggestions by directly referencing the context. Be concise but opinionated."
    )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"CONTEXT FROM REVIEWS:\n{context_str}\n\n"
        f"USER'S QUERY: '{query_text}'"
    )

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"Authorization": f"Bearer {auth_token}"}
    payload = {"contents": [{"parts": [{"text": full_prompt}]}]}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, headers=headers, timeout=60.0)
            response.raise_for_status()
            response_data = response.json()
            response_text = response_data['candidates'][0]['content']['parts'][0]['text']

        sources = [
            {
                "album_title": chunk['album_title'],
                "artist": chunk['artist'],
                "url": chunk['review_url']
            } for chunk in context_chunks
        ]
        unique_sources = [dict(t) for t in {tuple(d.items()) for d in sources}]

        return {"response": response_text, "sources": unique_sources}

    except Exception as e:
        error_message = f"An error occurred with the Gemini API: {e}"
        return {"response": error_message, "sources": []}


# --- API ENDPOINT ---
@app.post("/recommend", response_model=Recommendation)
async def get_recommendation(query: Query):
    """
    Main endpoint for recommendations, using the new tag-enriched database.
    """
    # 1. Perform a direct search. The DB is now smart enough.
    context = search_reviews(query.text, query.top_k)

    if not context:
        return {
            "response": "Apologies, but none of the reviews in my collection seem to match that particular vibe. Try a different query.",
            "sources": []
        }

    # 2. Generate a response with the retrieved context
    result = await generate_response(query.text, context)

    return result
