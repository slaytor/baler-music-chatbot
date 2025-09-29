import os
import httpx
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
MODEL_NAME = 'all-MiniLM-L6-v2'
DB_PATH = "./chroma_db"
COLLECTION_NAME = "pitchfork_reviews"
# --- THE FIX: Using the full, stable name for the latest flash model ---
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"

# You can set this in your terminal: export GOOGLE_API_KEY='your_key_here'
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# --- INITIALIZATION ---

# Load the embedding model
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")


# Connect to the ChromaDB database
print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
print("ChromaDB connection successful.")


# Initialize the FastAPI application
app = FastAPI(
    title="Baler Music Recommendation API",
    description="A chatbot for nuanced music recommendations from Pitchfork reviews.",
)


# --- API DATA MODELS ---

class Query(BaseModel):
    text: str
    top_k: int = 5


class Recommendation(BaseModel):
    response: str
    sources: list[dict]


# --- CORE RAG LOGIC ---

def search_reviews(query_text: str, top_k: int) -> list[dict]:
    """
    Searches the vector database for the most relevant review chunks.
    """
    query_embedding = model.encode([query_text])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Extract and format the results
    retrieved_chunks = []
    for i in range(len(results['ids'][0])):
        chunk_info = {
            "text": results['documents'][0][i],
            "metadata": results['metadatas'][0][i]
        }
        retrieved_chunks.append(chunk_info)

    return retrieved_chunks


def generate_response(query_text: str, context_chunks: list[dict]) -> dict:
    """
    Generates a response using a direct REST call to the Google Gemini API.
    """
    if not GOOGLE_API_KEY:
        return {"response": "Google API key is not configured.", "sources": []}

    context_str = "\n\n".join([f"From review of '{chunk['metadata']['album_title']}':\n{chunk['text']}" for chunk in context_chunks])

    system_prompt = (
        "You are a music recommendation assistant named Baler. Your knowledge is based "
        "solely on the context provided from Pitchfork reviews. Answer the user's query "
        "by synthesizing information from the provided context. Mention the album titles "
        "and artists. Do not use any outside knowledge."
    )

    # Combine system instructions and user query into a single prompt
    full_prompt = (
        f"{system_prompt}\n\n"
        f"Context from reviews:\n{context_str}\n\n"
        f"User's query:\n{query_text}"
    )

    # Using the v1beta API endpoint required for the latest models
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GOOGLE_API_KEY}"

    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}]
    }

    try:
        with httpx.Client() as client:
            response = client.post(api_url, json=payload, timeout=60.0)
            response.raise_for_status() # Will raise an exception for 4xx/5xx errors

            response_data = response.json()
            response_text = response_data['candidates'][0]['content']['parts'][0]['text']

        # Format sources for the final output
        sources = [
            {
                "album_title": chunk['metadata']['album_title'],
                "artist": chunk['metadata']['artist'],
                "url": chunk['metadata']['review_url']
            }
            for chunk in context_chunks
        ]
        unique_sources = [dict(t) for t in {tuple(d.items()) for d in sources}]

        return {"response": response_text, "sources": unique_sources}

    except httpx.HTTPStatusError as e:
        error_message = f"HTTP Error with Gemini API: {e.response.status_code} {e.response.text}"
        print(error_message)
        return {"response": error_message, "sources": []}
    except Exception as e:
        error_message = f"An error occurred with the Gemini API: {e}"
        print(error_message)
        return {"response": error_message, "sources": []}


# --- API ENDPOINT ---

@app.post("/recommend", response_model=Recommendation)
def get_recommendation(query: Query):
    """
    Main endpoint to get a music recommendation.
    """
    context = search_reviews(query.text, query.top_k)
    result = generate_response(query.text, context)
    return result
