import asyncio
import json
import time
from typing import AsyncGenerator

import httpx

from . import config
from .auth_util import get_gcp_auth_token


class GeminiClient:
    """Handles all API interactions with the Google Gemini model."""

    def __init__(self):
        self.api_url_base = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}"
        self.auth_token = None
        self.token_gen_time = 0
        self._get_token()  # Initial token fetch

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
        self._check_token()  # Ensure token is fresh before every call
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
                response = await client.post(
                    api_url, json=payload, headers=self._get_headers(), timeout=45.0
                )
                response.raise_for_status()
                response_data = response.json()
                if "candidates" in response_data and response_data["candidates"]:
                    content = response_data["candidates"][0]["content"]["parts"][0][
                        "text"
                    ]
                    json_str = content.strip().replace("```json", "").replace("```", "")
                    if not json_str:
                        return []
                    return json.loads(json_str)
                else:
                    return []
            except httpx.HTTPStatusError as e:
                if e.response.status_code in [401, 403]:  # Auth errors
                    print(f"Auth error: {e.response.text}. Attempting token refresh...")
                    self._get_token()  # Force refresh and retry
                    continue  # Try again with the new token
                elif e.response.status_code in [429, 503]:  # Rate limit/server errors
                    wait_time = (2**attempt) + 1
                    print(
                        f"API Error {e.response.status_code}. Retrying in {wait_time} seconds..."
                    )
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
            [
                f"From a review of '{chunk['album_title']}' by {chunk['artist']}:\n...{chunk['text_chunk']}..."
                for chunk in context_chunks
            ]
        )

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
        
        # --- FINAL DEBUGGING ---
        print("--- SENDING PROMPT TO GEMINI ---")
        print(full_prompt)
        print("---------------------------------")

        api_url = f"{self.api_url_base}:streamGenerateContent?alt=sse"
        payload = {"contents": [{"parts": [{"text": full_prompt}]}]}

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", api_url, json=payload, headers=self._get_headers(), timeout=60.0
                ) as response:
                    if response.status_code != 200:
                        error_content = await response.aread()
                        yield json.dumps(
                            {
                                "error": f"Gemini API Error {response.status_code}: {error_content.decode()}"
                            }
                        ) + "\n"
                        return

                    async for line in response.aiter_lines():
                        print(f"GEMINI STREAM LINE: {line}")
                        
                        if line.startswith("data:"):
                            try:
                                data_str = line[len("data:") :].strip()
                                data = json.loads(data_str)
                                if "candidates" in data and data["candidates"]:
                                    text_chunk = data["candidates"][0]["content"]["parts"][
                                        0
                                    ]["text"]
                                    yield json.dumps({"chunk": text_chunk}) + "\n"
                            except Exception as e:
                                yield json.dumps(
                                    {"error": f"Error processing stream: {e}"}
                                ) + "\n"

            # After the stream, send the sources
            sources = [
                {
                    "album_title": c["album_title"],
                    "artist": c["artist"],
                    "url": c["review_url"],
                }
                for c in context_chunks
            ]
            unique_sources = [dict(t) for t in {tuple(d.items()) for d in sources}]
            yield json.dumps({"sources": unique_sources}) + "\n"

        except Exception as e:
            yield json.dumps({"error": f"An error occurred streaming: {e}"}) + "\n"
