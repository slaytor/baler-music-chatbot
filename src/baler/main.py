import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .database import VectorDB
from .llm import get_llm_client # --- FIX: Import the factory function ---

# --- INITIALIZATION ---
try:
    db = VectorDB()
    llm = get_llm_client() # --- FIX: Use the factory to get the correct client ---
except Exception as e:
    print(f"FATAL: Failed to initialize services: {e}")
    exit(1)

app = FastAPI(
    title="Baler Music Recommendation API",
    description="A chatbot for nuanced music recommendations from Pitchfork reviews.",
)

# --- MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine the path to the static directory
static_dir = Path(__file__).parent.parent.parent / "static"

# --- API ENDPOINTS ---

class Query(BaseModel):
    text: str = Field(..., max_length=1000)
    top_k: int = 5

@app.post("/recommend")
async def get_recommendation_stream(query: Query):
    """
    Main endpoint for recommendations.
    Orchestrates the RAG process by calling modular services.
    """
    context = db.search(query.text, query.top_k)
    if not context:
        async def empty_stream():
            # Send the response in two parts to mimic the real stream
            yield json.dumps({"chunk": "Apologies, but my knowledge base is currently empty. Please run the data ingestion script to populate the database."}) + "\n"
            yield json.dumps({"sources": []}) + "\n"
        return StreamingResponse(empty_stream(), media_type="application/x-ndjson")
        
    return StreamingResponse(
        llm.stream_response(query.text, context), media_type="application/x-ndjson"
    )

# --- UI SERVING ---

@app.get("/")
async def read_index():
    """Serves the main index.html file."""
    return FileResponse(static_dir / "index.html")

# Mount the static directory to serve files like CSS, JS, etc.
app.mount("/static", StaticFiles(directory=static_dir), name="static")
