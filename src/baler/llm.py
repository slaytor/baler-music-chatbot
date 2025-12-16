import asyncio
import json
import time
from typing import AsyncGenerator

import httpx
# --- NEW: Import Google Auth libraries ---
import google.auth
import google.auth.transport.requests

from . import config

# --- CONSTANTS ---

SYSTEM_PROMPT = (
    "You are Baler, an AI music critic in the style of a Pitchfork reviewer. You are "
    "knowledgeable, a little bit pretentious, and have a distinctive voice. Your "
    "recommendations must be based ONLY on the provided review excerpts. "
    "IMPORTANT: When recommending an album, refer to it by the exact title provided in the context. "
    "If the review is for a compilation (Various Artists), recommend the compilation itself, "
    "but feel free to highlight specific artists mentioned within it as reasons to listen. "
    "Present one main recommendation, and one alternate choice."
)

# --- CLIENT FACTORY ---

def get_llm_client(provider: str = None):
    """
    Factory function to get the appropriate LLM client.
    If a provider is specified, it's used. Otherwise, it defaults to LLM_PROVIDER.
    """
    provider_to_use = provider or config.LLM_PROVIDER
    provider_upper = provider_to_use.upper()

    if provider_upper == "OLLAMA":
        return OllamaClient()
    elif provider_upper == "GEMINI":
        return GeminiClient()
    else:
        raise ValueError(f"Unknown LLM provider: {provider_to_use}")

# ----------------------

class OllamaClient:
    """Handles all API interactions with a local Ollama model."""

    def __init__(self):
        self.api_url = config.OLLAMA_API_URL
        self.model = config.OLLAMA_MODEL

    async def stream_response(
        self, query_text: str, context_chunks: list[dict]
    ) -> AsyncGenerator[str, None]:
        """Streams a chat response from Ollama based on the query and context."""
        context_str = "\n\n".join(
            [f"From a review of '{c['album_title']}' by {c['artist']}:\n...{c['text_chunk']}..." for c in context_chunks]
        )
        
        full_prompt = f"CONTEXT FROM REVIEWS:\n{context_str}\n\nUSER'S QUERY: '{query_text}'"

        payload = {"model": self.model, "system": SYSTEM_PROMPT, "prompt": full_prompt, "stream": True}

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", self.api_url, json=payload, timeout=300.0) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if not data.get("done"):
                                yield json.dumps({"chunk": data.get("response", "")}) + "\n"
            
            sources = [
                {
                    "album_title": c["album_title"],
                    "artist": c["artist"],
                    "url": c["review_url"],
                    "album_cover_url": c.get("album_cover_url", "N/A"),
                    "score": c.get("score", "N/A")
                }
                for c in context_chunks
            ]
            unique_sources = [dict(t) for t in {tuple(d.items()) for d in sources}]
            yield json.dumps({"sources": unique_sources}) + "\n"

        except httpx.RequestError as e:
            yield json.dumps({"error": f"Could not connect to Ollama. Please ensure it's running. Details: {e!r}"}) + "\n"
        except Exception as e:
            yield json.dumps({"error": f"An error occurred streaming from Ollama: {e}"}) + "\n"


class GeminiClient:
    """Handles all API interactions with the Google Gemini model."""

    def __init__(self):
        self.api_url_base = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}"
        # --- NEW: Use google-auth to handle credentials automatically ---
        self.credentials, self.project_id = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    def _get_auth_headers(self) -> dict:
        """Gets valid authentication headers, refreshing the token if needed."""
        # The google-auth library handles caching and refreshing automatically.
        auth_req = google.auth.transport.requests.Request()
        self.credentials.refresh(auth_req)
        return {"Authorization": f"Bearer {self.credentials.token}"}

    async def stream_response(
        self, query_text: str, context_chunks: list[dict]
    ) -> AsyncGenerator[str, None]:
        """Streams a chat response from Gemini based on the query and context."""
        context_str = "\n\n".join(
            [f"From a review of '{c['album_title']}' by {c['artist']}:\n...{c['text_chunk']}..." for c in context_chunks]
        )

        full_prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT FROM REVIEWS:\n{context_str}\n\nUSER'S QUERY: '{query_text}'"
        
        api_url = f"{self.api_url_base}:streamGenerateContent?alt=sse"
        payload = {"contents": [{"parts": [{"text": full_prompt}]}]}

        try:
            headers = self._get_auth_headers()
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", api_url, json=payload, headers=headers, timeout=60.0) as response:
                    if response.status_code != 200:
                        error_content = await response.aread()
                        yield json.dumps({"error": f"Gemini API Error {response.status_code}: {error_content.decode()}"}) + "\n"
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            try:
                                data_str = line[len("data:") :].strip()
                                data = json.loads(data_str)
                                if "candidates" in data and data["candidates"]:
                                    yield json.dumps({"chunk": data["candidates"][0]["content"]["parts"][0]["text"]}) + "\n"
                            except Exception as e:
                                yield json.dumps({"error": f"Error processing stream: {e}"}) + "\n"

            sources = [
                {
                    "album_title": c["album_title"],
                    "artist": c["artist"],
                    "url": c["review_url"],
                    "album_cover_url": c.get("album_cover_url", "N/A"),
                    "score": c.get("score", "N/A")
                }
                for c in context_chunks
            ]
            unique_sources = [dict(t) for t in {tuple(d.items()) for d in sources}]
            yield json.dumps({"sources": unique_sources}) + "\n"

        except Exception as e:
            yield json.dumps({"error": f"An error occurred streaming: {e}"}) + "\n"
