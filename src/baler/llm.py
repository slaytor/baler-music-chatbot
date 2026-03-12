import json
import re
from collections.abc import AsyncGenerator

import google.auth
import google.auth.transport.requests
import httpx

from . import config
from .utils import parse_json_list

# --- CONSTANTS ---

SYSTEM_PROMPT = (
    "You are Baler, an AI music critic in the style of a Pitchfork reviewer. You are "
    "knowledgeable, a little bit pretentious, and have a distinctive voice. Your "
    "recommendations must be based ONLY on the provided review excerpts. "
    "IMPORTANT: When recommending an album, refer to it by the exact title provided in the context. "
    "If the review is for a compilation (Various Artists), recommend the compilation itself, "
    "but feel free to highlight specific artists mentioned within it as reasons to listen. "
    "Each album entry includes genre tags and related artists — use these to draw sharp, "
    "specific connections between recommendations and the user's taste. Mention genres and "
    "sonic relationships where they illuminate why an album fits the query. "
    "Present one main recommendation, and one second choice. For each, explain specifically why it fits the query."
)

TAG_PROMPT = (
    "Generate 5-8 descriptive tags for the following music review excerpt. "
    "{genre_context}"
    "Focus on mood/atmosphere, production style, instrumentation, and sonic characteristics. "
    "Return ONLY a valid JSON array of lowercase strings. "
    'Example: ["melancholic", "lo-fi", "tape-saturated", "guitar-driven", "claustrophobic"]\n\n'
    "Review excerpt: {chunk}"
)

_TAG_GENRE_CONTEXT = "The artist's genres are already known ({genres}), so avoid simply repeating them — "
_TAG_NO_GENRE_CONTEXT = "Tags should cover genre, subgenre, mood/atmosphere, production style, and instrumentation. "

FILTER_EXTRACTION_PROMPT = (
    "Analyze this music recommendation query and extract any exclusion filters. "
    "Return a JSON object with exactly these fields:\n"
    '- "clean_query": the query with all exclusion/negative language removed (what the user WANTS)\n'
    '- "exclude_genres": list of music genres to exclude, lowercase (e.g. ["hip-hop", "jazz"])\n'
    '- "exclude_artists": list of artist names to exclude\n'
    '- "max_year": integer or null — exclude albums released after this year ("nothing after 2010" → 2010)\n'
    '- "min_year": integer or null — exclude albums released before this year ("only post-1990" → 1990)\n'
    "If there are no exclusions, return empty lists and null for years. Return ONLY valid JSON.\n\n"
    "Query: {query}"
)

_FILTER_DEFAULTS = {
    "clean_query": None,
    "exclude_genres": [],
    "exclude_artists": [],
    "max_year": None,
    "min_year": None,
}


def _format_context_entry(c: dict) -> str:
    """Format a single context chunk with structured metadata for the LLM."""
    genres = parse_json_list(c.get("artist_genres"))
    related = parse_json_list(c.get("related_artists"))
    lines = [f"Album: '{c['album_title']}' by {c['artist']}"]
    if genres:
        lines.append(f"Genres: {', '.join(genres)}")
    if related:
        lines.append(f"Related artists: {', '.join(related[:5])}")
    lines.append(f"Review excerpt: ...{c['text_chunk']}...")
    return "\n".join(lines)


def _format_sources(context_chunks: list[dict]) -> list[dict]:
    """Build the unique sources list from context chunks for the NDJSON stream."""
    sources = [
        {
            "album_title": c["album_title"],
            "artist": c["artist"],
            "url": c["review_url"],
            "album_cover_url": c.get("album_cover_url", "N/A"),
            "score": c.get("score", "N/A"),
        }
        for c in context_chunks
    ]
    return [dict(t) for t in {tuple(d.items()) for d in sources}]


def _parse_tags(text: str) -> list[str]:
    """Extract a JSON array of tags from an LLM response, tolerating prose wrapping."""
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return []
    try:
        tags = json.loads(match.group())
        return [t.lower().strip() for t in tags if isinstance(t, str)]
    except json.JSONDecodeError:
        return []


# --- CLIENT FACTORY ---

def get_llm_client(provider: str = None):
    """Factory function to get the appropriate LLM client."""
    provider_to_use = provider or config.LLM_PROVIDER
    provider_upper = provider_to_use.upper()

    if provider_upper == "OLLAMA":
        return OllamaClient()
    elif provider_upper == "GEMINI":
        return GeminiClient()
    elif provider_upper == "VERTEX":
        return VertexClient()
    else:
        raise ValueError(f"Unknown LLM provider: {provider_to_use}")


# --- SHARED AUTH HELPER ---

def _get_google_credentials():
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    return {"Authorization": f"Bearer {credentials.token}"}


# --- CLIENTS ---

class OllamaClient:
    """Handles all API interactions with a local Ollama model."""

    def __init__(self):
        self.api_url = config.OLLAMA_API_URL
        self.model = config.OLLAMA_MODEL

    async def generate_tags_for_chunk(self, client: httpx.AsyncClient, chunk: str, genres: list[str] | None = None) -> list[str]:
        """Generates semantic tags for a review chunk using Ollama."""
        genre_context = _TAG_GENRE_CONTEXT.format(genres=", ".join(genres)) if genres else _TAG_NO_GENRE_CONTEXT
        prompt = TAG_PROMPT.format(chunk=chunk, genre_context=genre_context)
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            response = await client.post(self.api_url, json=payload, timeout=60.0)
            response.raise_for_status()
            text = response.json().get("response", "")
            return _parse_tags(text)
        except Exception:
            return []

    async def extract_filters(self, query: str) -> dict:
        """Ollama stub — filter extraction not supported, returns no filters."""
        return {**_FILTER_DEFAULTS, "clean_query": query}

    async def stream_response(
        self, query_text: str, context_chunks: list[dict]
    ) -> AsyncGenerator[str, None]:
        """Streams a chat response from Ollama based on the query and context."""
        context_str = "\n\n".join(_format_context_entry(c) for c in context_chunks)
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

            yield json.dumps({"sources": _format_sources(context_chunks)}) + "\n"

        except httpx.RequestError as e:
            yield json.dumps({"error": f"Could not connect to Ollama. Please ensure it's running. Details: {e!r}"}) + "\n"
        except Exception as e:
            yield json.dumps({"error": f"An error occurred streaming from Ollama: {e}"}) + "\n"


class GeminiClient:
    """Handles all API interactions with the Google Gemini free-tier API (used for app inference)."""

    def __init__(self):
        self.api_url_base = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}"
        self.credentials, _ = google.auth.default(
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/generative-language",
            ]
        )

    def _get_auth_headers(self) -> dict:
        auth_req = google.auth.transport.requests.Request()
        self.credentials.refresh(auth_req)
        return {"Authorization": f"Bearer {self.credentials.token}"}

    async def generate_tags_for_chunk(self, client: httpx.AsyncClient, chunk: str, genres: list[str] | None = None) -> list[str]:
        """Generates semantic tags for a review chunk using Gemini."""
        genre_context = _TAG_GENRE_CONTEXT.format(genres=", ".join(genres)) if genres else _TAG_NO_GENRE_CONTEXT
        prompt = TAG_PROMPT.format(chunk=chunk, genre_context=genre_context)
        api_url = f"{self.api_url_base}:generateContent"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            headers = self._get_auth_headers()
            response = await client.post(api_url, json=payload, headers=headers, timeout=30.0)
            response.raise_for_status()
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return _parse_tags(text)
        except Exception:
            return []

    async def extract_filters(self, query: str) -> dict:
        """Parses exclusion filters from a user query using a fast Gemini call."""
        default = {**_FILTER_DEFAULTS, "clean_query": query}
        prompt = FILTER_EXTRACTION_PROMPT.format(query=query)
        api_url = f"{self.api_url_base}:generateContent"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            async with httpx.AsyncClient() as client:
                headers = self._get_auth_headers()
                response = await client.post(api_url, json=payload, headers=headers, timeout=10.0)
                response.raise_for_status()
                text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return default
            parsed = json.loads(match.group())
            return {**default, **parsed}
        except Exception:
            return default

    async def stream_response(
        self, query_text: str, context_chunks: list[dict]
    ) -> AsyncGenerator[str, None]:
        """Streams a chat response from Gemini based on the query and context."""
        context_str = "\n\n".join(_format_context_entry(c) for c in context_chunks)
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
                                data = json.loads(line[len("data:"):].strip())
                                if "candidates" in data and data["candidates"]:
                                    yield json.dumps({"chunk": data["candidates"][0]["content"]["parts"][0]["text"]}) + "\n"
                            except Exception as e:
                                yield json.dumps({"error": f"Error processing stream: {e}"}) + "\n"

            yield json.dumps({"sources": _format_sources(context_chunks)}) + "\n"

        except Exception as e:
            yield json.dumps({"error": f"An error occurred streaming: {e}"}) + "\n"


class VertexClient:
    """
    Handles Gemini via Vertex AI (Google Cloud) for KB ingestion.
    Uses the same model as GeminiClient but billed through GCP with no daily quota.
    """

    def __init__(self):
        project = config.GCP_PROJECT_ID
        region = config.GCP_REGION
        model = config.GEMINI_MODEL
        self.api_url_base = (
            f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/{region}/publishers/google/models/{model}"
        )

    def _get_auth_headers(self) -> dict:
        return _get_google_credentials()

    async def generate_tags_for_chunk(self, client: httpx.AsyncClient, chunk: str, genres: list[str] | None = None) -> list[str]:
        """Generates semantic tags for a review chunk using Vertex AI."""
        genre_context = _TAG_GENRE_CONTEXT.format(genres=", ".join(genres)) if genres else _TAG_NO_GENRE_CONTEXT
        prompt = TAG_PROMPT.format(chunk=chunk, genre_context=genre_context)
        api_url = f"{self.api_url_base}:generateContent"
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        try:
            headers = self._get_auth_headers()
            response = await client.post(api_url, json=payload, headers=headers, timeout=30.0)
            response.raise_for_status()
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return _parse_tags(text)
        except Exception:
            return []

    async def extract_filters(self, query: str) -> dict:
        """Vertex stub — filter extraction not used in production (app uses GeminiClient)."""
        return {**_FILTER_DEFAULTS, "clean_query": query}

    async def stream_response(
        self, query_text: str, context_chunks: list[dict]
    ) -> AsyncGenerator[str, None]:
        """Streams a chat response from Vertex AI (for local testing with VERTEX provider)."""
        context_str = "\n\n".join(_format_context_entry(c) for c in context_chunks)
        full_prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT FROM REVIEWS:\n{context_str}\n\nUSER'S QUERY: '{query_text}'"

        api_url = f"{self.api_url_base}:streamGenerateContent?alt=sse"
        payload = {"contents": [{"role": "user", "parts": [{"text": full_prompt}]}]}

        try:
            headers = self._get_auth_headers()
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", api_url, json=payload, headers=headers, timeout=60.0) as response:
                    if response.status_code != 200:
                        error_content = await response.aread()
                        yield json.dumps({"error": f"Vertex API Error {response.status_code}: {error_content.decode()}"}) + "\n"
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            try:
                                data = json.loads(line[len("data:"):].strip())
                                if "candidates" in data and data["candidates"]:
                                    yield json.dumps({"chunk": data["candidates"][0]["content"]["parts"][0]["text"]}) + "\n"
                            except Exception as e:
                                yield json.dumps({"error": f"Error processing stream: {e}"}) + "\n"

            yield json.dumps({"sources": _format_sources(context_chunks)}) + "\n"

        except Exception as e:
            yield json.dumps({"error": f"An error occurred streaming: {e}"}) + "\n"
