import asyncio
import json
import time
from typing import AsyncGenerator, Union

import httpx

from . import config
from .auth_util import get_gcp_auth_token

# --- UTILITY FUNCTION ---

def extract_json_from_string(text: str) -> str:
    """
    Finds and extracts the first valid JSON object (list or dict) from a string.
    """
    start_bracket = text.find('[')
    start_brace = text.find('{')

    if start_bracket == -1 and start_brace == -1: return ""
    
    start_index = min(start_bracket, start_brace) if start_bracket != -1 and start_brace != -1 else max(start_bracket, start_brace)
    
    end_bracket = text.rfind(']')
    end_brace = text.rfind('}')

    if end_bracket == -1 and end_brace == -1: return ""

    end_index = max(end_bracket, end_brace)

    if start_index > end_index: return ""

    return text[start_index : end_index + 1]

# --- CLIENT FACTORY ---

def get_llm_client():
    """Factory function to get the appropriate LLM client based on config."""
    provider = config.LLM_PROVIDER.upper()
    if provider == "OLLAMA":
        return OllamaClient()
    elif provider == "GEMINI":
        return GeminiClient()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")

# ----------------------

class OllamaClient:
    """Handles all API interactions with a local Ollama model."""

    def __init__(self):
        self.api_url = config.OLLAMA_API_URL
        self.model = config.OLLAMA_MODEL

    async def generate_tags_for_chunk(
        self, client: httpx.AsyncClient, chunk: str
    ) -> list[str]:
        """Uses a local Ollama model to extract descriptive tags from a text chunk."""
        prompt = (
            "You are an expert musicologist. Analyze the following excerpt from a music review. "
            "Extract 5-7 descriptive keywords and phrases that capture the mood, genre, "
            "and instrumentation.\n\n"
            f'REVIEW EXCERPT:\n"...{chunk}..."\n\n'
            "Your response MUST be a single line of comma-separated values. Do not use JSON. "
            "For example: dreamy, shimmering guitars, hazy atmosphere, introspective"
        )

        payload = {"model": self.model, "prompt": prompt, "stream": False}

        try:
            response = await client.post(self.api_url, json=payload, timeout=300.0)
            response.raise_for_status()
            response_data = response.json()
            
            raw_output = response_data.get("response", "")
            if not raw_output:
                return []

            return [tag.strip() for tag in raw_output.split(',') if tag.strip()]

        except httpx.RequestError as e:
            print(f"\n--- OLLAMA CONNECTION ERROR ---\nCould not connect to Ollama at {self.api_url}\nPlease ensure the Ollama application is running.\nError details: {e!r}\n---------------------------------\n")
            return []
        except Exception as e:
            print(f"An unexpected error occurred in OllamaClient: {type(e).__name__}: {e}")
            return []

    async def stream_response(
        self, query_text: str, context_chunks: list[dict]
    ) -> AsyncGenerator[str, None]:
        """Streams a chat response from Ollama based on the query and context."""
        context_str = "\n\n".join(
            [f"From a review of '{c['album_title']}' by {c['artist']}:\n...{c['text_chunk']}..." for c in context_chunks]
        )

        system_prompt = (
            "You are Baler, an AI music critic in the style of a Pitchfork reviewer. You are "
            "knowledgeable, a little bit pretentious, and have a distinctive voice. Your "
            "recommendations must be based ONLY on the provided review excerpts. Justify your "
            "suggestions by directly referencing the context. Be concise but opinionated."
        )
        
        full_prompt = f"CONTEXT FROM REVIEWS:\n{context_str}\n\nUSER'S QUERY: '{query_text}'"

        payload = {"model": self.model, "system": system_prompt, "prompt": full_prompt, "stream": True}

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", self.api_url, json=payload, timeout=300.0) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if not data.get("done"):
                                yield json.dumps({"chunk": data.get("response", "")}) + "\n"
            
            sources = [dict(t) for t in {tuple(d.items()) for d in [{"album_title": c["album_title"], "artist": c["artist"], "url": c["review_url"]} for c in context_chunks]}]
            yield json.dumps({"sources": sources}) + "\n"

        except httpx.RequestError as e:
            yield json.dumps({"error": f"Could not connect to Ollama. Please ensure it's running. Details: {e!r}"}) + "\n"
        except Exception as e:
            yield json.dumps({"error": f"An error occurred streaming from Ollama: {e}"}) + "\n"


class GeminiClient:
    """Handles all API interactions with the Google Gemini model."""

    def __init__(self):
        self.api_url_base = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}"
        self.auth_token = None
        self.token_gen_time = 0
        self._get_token()

    def _get_token(self):
        """Fetches or refreshes the GCP auth token."""
        print("Getting/Refreshing GCP Authentication Token...")
        self.auth_token = get_gcp_auth_token()
        self.token_gen_time = time.time()
        if not self.auth_token:
            raise Exception("FATAL: Could not get GCP auth token.")
        print("Token refreshed successfully.")

    def _check_token(self):
        """Checks if the token is expired and refreshes it if needed."""
        if time.time() - self.token_gen_time > config.TOKEN_LIFESPAN_SECONDS:
            self._get_token()

    def _get_headers(self) -> dict:
        """Returns the required authentication headers."""
        self._check_token()
        return {"Authorization": f"Bearer {self.auth_token}"}

    async def generate_tags_for_chunk(
        self, client: httpx.AsyncClient, chunk: str
    ) -> list[str]:
        """Uses Gemini to extract descriptive tags from a text chunk, with retries."""
        prompt = (
            "You are an expert musicologist. Analyze the following excerpt from a music review. "
            "Extract a list of 5-7 descriptive keywords and phrases that capture the mood, genre, "
            "instrumentation, and overall sonic texture. Focus on evocative adjectives.\n\n"
            f'REVIEW EXCERPT:\n"...{chunk}..."\n\n'
            "Return ONLY a JSON list of strings. For example: "
            '["dream-pop", "shimmering guitars", "hazy atmosphere", "ethereal vocals", "introspective"]'
        )

        api_url = f"{self.api_url_base}:generateContent"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        for attempt in range(5):
            try:
                response = await client.post(api_url, json=payload, headers=self._get_headers(), timeout=45.0)
                response.raise_for_status()
                response_data = response.json()
                if "candidates" in response_data and response_data["candidates"]:
                    content = response_data["candidates"][0]["content"]["parts"][0]["text"]
                    json_str = content.strip().replace("```json", "").replace("```", "")
                    if not json_str: return []
                    return json.loads(json_str)
                else:
                    return []
            except httpx.HTTPStatusError as e:
                if e.response.status_code in [401, 403]:
                    print(f"Auth error: {e.response.text}. Attempting token refresh...")
                    self._get_token()
                    continue
                elif e.response.status_code in [429, 503]:
                    wait_time = (2**attempt) + 1
                    print(f"API Error {e.response.status_code}. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Non-retryable HTTP Error: {e.response.text}")
                    return []
            except Exception as e:
                print(f"Error generating tags: {e}")
                return []
        print("API is still unavailable after multiple retries. Skipping chunk.")
        return []

    async def stream_response(
        self, query_text: str, context_chunks: list[dict]
    ) -> AsyncGenerator[str, None]:
        """Streams a chat response from Gemini based on the query and context."""
        context_str = "\n\n".join(
            [f"From a review of '{c['album_title']}' by {c['artist']}:\n...{c['text_chunk']}..." for c in context_chunks]
        )

        system_prompt = (
            "You are Baler, an AI music critic in the style of a Pitchfork reviewer. You are "
            "knowledgeable, a little bit pretentious, and have a distinctive voice. Your "
            "recommendations must be based ONLY on the provided review excerpts. Justify your "
            "suggestions by directly referencing the context. Be concise but opinionated."
        )

        full_prompt = f"{system_prompt}\n\nCONTEXT FROM REVIEWS:\n{context_str}\n\nUSER'S QUERY: '{query_text}'"
        
        api_url = f"{self.api_url_base}:streamGenerateContent?alt=sse"
        payload = {"contents": [{"parts": [{"text": full_prompt}]}]}

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", api_url, json=payload, headers=self._get_headers(), timeout=60.0) as response:
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

            sources = [dict(t) for t in {tuple(d.items()) for d in [{"album_title": c["album_title"], "artist": c["artist"], "url": c["review_url"]} for c in context_chunks]}]
            yield json.dumps({"sources": sources}) + "\n"

        except Exception as e:
            yield json.dumps({"error": f"An error occurred streaming: {e}"}) + "\n"
