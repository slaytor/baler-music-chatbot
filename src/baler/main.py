import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .database import VectorDB
from .llm import get_llm_client
from .music_services import SpotifyClient

# --- INITIALIZATION ---
try:
    db = VectorDB()
    llm = get_llm_client(provider=config.APP_LLM_PROVIDER)
    spotify = SpotifyClient()
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

# --- API MODELS ---

class Query(BaseModel):
    text: str = Field(..., max_length=1000)
    top_k: int = 2

class AlbumQuery(BaseModel):
    album_title: str
    artist: str

# --- API ENDPOINTS ---

@app.post("/recommend")
async def get_recommendation_stream(query: Query):
    """
    Main endpoint for recommendations.
    Orchestrates the RAG process by calling modular services.
    """
    # --- CHANGE: Reduced fetch count to 20 ---
    FETCH_COUNT = 20
    raw_matches = db.search(query.text, FETCH_COUNT)
    
    if not raw_matches:
        async def empty_stream():
            yield json.dumps({"chunk": "Apologies, but my knowledge base is currently empty. Please run the data ingestion script to populate the database."}) + "\n"
            yield json.dumps({"sources": []}) + "\n"
        return StreamingResponse(empty_stream(), media_type="application/x-ndjson")
    
    # Deduplicate matches based on review_url
    unique_matches = []
    seen_urls = set()
    for match in raw_matches:
        url = match.get("review_url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_matches.append(match)

    # The first 2 are for the LLM context
    context = unique_matches[:query.top_k]
    
    # The rest are for the "Show more" feature
    remaining_matches = unique_matches[query.top_k:]
    
    # Format remaining matches for the frontend
    formatted_remaining = [
        {
            "album_title": c["album_title"],
            "artist": c["artist"],
            "url": c["review_url"],
            "album_cover_url": c.get("album_cover_url", "N/A"),
            "score": c.get("score", "N/A")
        }
        for c in remaining_matches
    ]
    
    # We need to inject the remaining matches into the stream.
    # We'll do this by modifying the generator to yield them at the end.
    
    async def stream_with_extras():
        # 1. Stream the LLM response as normal
        async for chunk in llm.stream_response(query.text, context):
            yield chunk
            
        # 2. After the LLM is done, yield the remaining matches as a special event
        yield json.dumps({"remaining_sources": formatted_remaining}) + "\n"

    return StreamingResponse(
        stream_with_extras(), media_type="application/x-ndjson"
    )

@app.post("/find-album-url")
async def find_album_url(album_query: AlbumQuery):
    """
    Finds an album's public URL on Spotify.
    """
    album_url = await spotify.get_album_spotify_url(
        album_title=album_query.album_title,
        artist=album_query.artist
    )
    if not album_url:
        return JSONResponse(
            status_code=404,
            content={"error": "Album not found on Spotify."}
        )
    return {"album_url": album_url}

# --- UI SERVING ---

@app.get("/")
async def read_index():
    """Serves the main index.html file."""
    return FileResponse(static_dir / "index.html")

# Mount the static directory to serve files like CSS, JS, etc.
app.mount("/static", StaticFiles(directory=static_dir), name="static")
