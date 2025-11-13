import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from .database import VectorDB
from .llm import GeminiClient

# --- INITIALIZATION ---
try:
    db = VectorDB()
    llm = GeminiClient()
except Exception as e:
    print(f"FATAL: Failed to initialize services: {e}")
    # In a real app, we'd want a more graceful failure
    # but for this, we'll exit if services can't start.
    exit(1)

app = FastAPI(
    title="Baler Music Recommendation API",
    description="A chatbot for nuanced music recommendations from Pitchfork reviews.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    text: str
    top_k: int = 5


@app.post("/recommend")
async def get_recommendation_stream(query: Query):
    """
    Main endpoint for recommendations.
    Orchestrates the RAG process by calling modular services.
    """

    # 1. Search (Retrieve)
    context = db.search(query.text, query.top_k)

    if not context:
         async def empty_stream():
             yield json.dumps({
                 "response": "Apologies, but none of the reviews in my collection seem to match that particular vibe. Try a different query.",
                 "sources": []
             }) + "\n"
         return StreamingResponse(empty_stream(), media_type="application/x-ndjson")

    # 2. Augment & Generate (Stream response)
    return StreamingResponse(
        llm.stream_response(query.text, context),
        media_type="application/x-ndjson"
    )
