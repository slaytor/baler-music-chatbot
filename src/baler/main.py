import os
import httpx
import json
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Any, AsyncGenerator
from dotenv import load_dotenv
from pathlib import Path
import asyncio

# Import the shared authentication function
from .auth_util import get_gcp_auth_token

# Load environment variables from .env file
load_dotenv()


# Remember your IDE's formatting preference for blank lines
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

# CORS Middleware (as before)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- API DATA MODELS ---
class Query(BaseModel):
    text: str
    top_k: int = 5

# No response model for streaming, we'll construct JSON manually


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

    # Check if 'metadatas' is present and has results
    if not results or not results.get('metadatas') or not results['metadatas'][0]:
        return [] # Return empty list if no results

    return [metadata for metadata in results['metadatas'][0]]


async def stream_gemini_response(query_text: str, context_chunks: list[dict]) -> AsyncGenerator[str, None]:
    """
    Streams the response from the Google Gemini API. Yields JSON strings for stream events.
    """
    auth_token = get_gcp_auth_token()
    if not auth_token:
        yield json.dumps({"error": "Could not authenticate with Google Cloud."}) + "\n"
        return

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

    # --- THE FIX: Use the streaming endpoint ---
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent?alt=sse"
    headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": full_prompt}]}]}

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", api_url, json=payload, headers=headers, timeout=60.0) as response:
                if response.status_code != 200:
                     error_content = await response.aread()
                     yield json.dumps({"error": f"Gemini API Error {response.status_code}: {error_content.decode()}"}) + "\n"
                     return

                # Process the Server-Sent Events (SSE) stream
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        try:
                            data_str = line[len("data:"):].strip()
                            data = json.loads(data_str)
                            # Extract the text chunk from the response
                            if 'candidates' in data and data['candidates'] and \
                               'content' in data['candidates'][0] and \
                               'parts' in data['candidates'][0]['content'] and \
                               data['candidates'][0]['content']['parts']:
                                text_chunk = data['candidates'][0]['content']['parts'][0]['text']
                                # Yield a JSON object representing the text chunk
                                yield json.dumps({"chunk": text_chunk}) + "\n"
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON line: {line}")
                        except Exception as e:
                            print(f"Error processing stream line: {e}")
                            yield json.dumps({"error": f"Error processing stream: {e}"}) + "\n"

        # After the stream, send the sources
        sources = [
            {
                "album_title": chunk['album_title'],
                "artist": chunk['artist'],
                "url": chunk['review_url']
            } for chunk in context_chunks
        ]
        unique_sources = [dict(t) for t in {tuple(d.items()) for d in sources}]
        yield json.dumps({"sources": unique_sources}) + "\n"

    except Exception as e:
        error_message = f"An error occurred streaming from Gemini API: {e}"
        print(error_message)
        yield json.dumps({"error": error_message}) + "\n"


# --- API ENDPOINT ---
@app.post("/recommend") # Removed response_model for streaming
async def get_recommendation_stream(query: Query):
    """
    Main endpoint for recommendations, streams the response.
    """
    context = search_reviews(query.text, query.top_k)

    if not context:
         # Still return a JSON structure even for errors/no context
         async def empty_stream():
             yield json.dumps({
                 "response": "Apologies, but none of the reviews in my collection seem to match that particular vibe. Try a different query.",
                 "sources": []
             }) + "\n"
         return StreamingResponse(empty_stream(), media_type="application/x-ndjson")

    # Return a streaming response
    return StreamingResponse(stream_gemini_response(query.text, context), media_type="application/x-ndjson")
