import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .database import VectorDB
from .llm import get_llm_client
from .music_services import SpotifyClient

# --- LOGGING SETUP ---
# Create a dedicated logger for user queries
query_logger = logging.getLogger("user_queries")
query_logger.setLevel(logging.INFO)

# 1. Console Handler (For Cloud/Docker logs)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(message)s"))
query_logger.addHandler(console_handler)

# 2. File Handler (For local persistent storage)
# We use the project root to store the log file
log_file_path = config.PROJECT_ROOT / "user_queries.log"
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(logging.Formatter("%(message)s"))
query_logger.addHandler(file_handler)

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
async def get_recommendation_stream(query: Query, request: Request):
    """
    Main endpoint for recommendations.
    Orchestrates the RAG process by calling modular services.
    """
    # --- LOGGING ---
    # Capture the query metadata
    log_entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "query": query.text,
        "top_k": query.top_k,
        # Attempt to get client IP (useful for distinguishing users, though often proxied)
        "client_ip": request.client.host if request.client else "unknown",
    }
    # Log as a JSON string for easy parsing later
    query_logger.info(json.dumps(log_entry))
    # ----------------

    # Extract exclusion filters and clean query from user input
    filters = await llm.extract_filters(query.text)
    search_query = filters.get("clean_query") or query.text

    # Hybrid retrieval: BM25 + vector search fused via RRF, then cross-encoder rerank
    candidates = db.hybrid_search(search_query, top_k=75)
    candidates = db.apply_exclusion_filters(candidates, filters)
    unique_matches = db.rerank(search_query, candidates)

    # Expand candidate pool with albums by related artists of top results
    expanded = db.expand_with_related_artists(search_query, unique_matches[:5])
    if expanded:
        expanded = db.apply_exclusion_filters(expanded, filters)
        expanded_reranked = db.rerank(search_query, expanded)
        existing_urls = {m.get("review_url") for m in unique_matches}
        for m in expanded_reranked:
            if m.get("review_url") not in existing_urls:
                unique_matches.append(m)
                existing_urls.add(m.get("review_url"))

    # Filter out albums by artists the user explicitly mentioned — they want novel discoveries
    query_lower = query.text.lower()
    unique_matches = [
        m for m in unique_matches if m.get("artist", "").lower() not in query_lower
    ]

    # Filter out albums where name-token overlap with the query is the likely reason
    # they surfaced (e.g. "Lotus Eater" ranking for "Flying Lotus" due to shared "lotus").
    # Only meaningful tokens (5+ chars, not generic music/query words) are checked.
    _GENERIC_TOKENS = {
        "about",
        "after",
        "again",
        "album",
        "along",
        "around",
        "based",
        "bring",
        "could",
        "every",
        "feels",
        "genre",
        "going",
        "great",
        "group",
        "heavy",
        "indie",
        "known",
        "label",
        "large",
        "later",
        "listen",
        "looking",
        "music",
        "other",
        "people",
        "quite",
        "really",
        "releases",
        "since",
        "something",
        "sound",
        "their",
        "there",
        "these",
        "those",
        "through",
        "times",
        "track",
        "where",
        "which",
        "while",
        "would",
    }
    query_tokens = {
        t for t in re.findall(r"\b[a-z]{5,}\b", query_lower) if t not in _GENERIC_TOKENS
    }
    if query_tokens:

        def _has_name_overlap(m: dict) -> bool:
            artist_tok = set(re.findall(r"\b[a-z]{5,}\b", m.get("artist", "").lower()))
            title_tok = set(
                re.findall(r"\b[a-z]{5,}\b", m.get("album_title", "").lower())
            )
            return bool(query_tokens & (artist_tok | title_tok))

        unique_matches = [m for m in unique_matches if not _has_name_overlap(m)]

    if not unique_matches:

        async def empty_stream():
            yield (
                json.dumps(
                    {
                        "chunk": "Apologies, but my knowledge base is currently empty. Please run the data ingestion script to populate the database."
                    }
                )
                + "\n"
            )
            yield json.dumps({"sources": []}) + "\n"

        return StreamingResponse(empty_stream(), media_type="application/x-ndjson")

    # The first top_k are for the LLM context
    context = unique_matches[: query.top_k]

    # The rest are for the "Show more" feature
    remaining_matches = unique_matches[query.top_k :]

    # Format remaining matches for the frontend
    formatted_remaining = [
        {
            "album_title": c["album_title"],
            "artist": c["artist"],
            "url": c["review_url"],
            "album_cover_url": c.get("album_cover_url", "N/A"),
            "score": c.get("score", "N/A"),
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

    return StreamingResponse(stream_with_extras(), media_type="application/x-ndjson")


@app.post("/find-album-url")
async def find_album_url(album_query: AlbumQuery):
    """
    Finds an album's public URL on Spotify.
    """
    album_url = await spotify.get_album_spotify_url(
        album_title=album_query.album_title, artist=album_query.artist
    )
    if not album_url:
        return JSONResponse(
            status_code=404, content={"error": "Album not found on Spotify."}
        )
    return {"album_url": album_url}


# --- UI SERVING ---


@app.get("/")
async def read_index():
    """Serves the main index.html file."""
    return FileResponse(static_dir / "index.html")


# Mount the static directory to serve files like CSS, JS, etc.
app.mount("/static", StaticFiles(directory=static_dir), name="static")
