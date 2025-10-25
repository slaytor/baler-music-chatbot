import os
import httpx
import json
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
import time
import asyncio
from dotenv import load_dotenv
from pathlib import Path

# Import the shared authentication function
from .auth_util import get_gcp_auth_token

# Load environment variables from .env file
load_dotenv()


# --- CONFIGURATION ---
MODEL_NAME = 'all-MiniLM-L6-v2'
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "chroma_db"
RAW_DATA_FILE = PROJECT_ROOT / "reviews.jsonl"
COLLECTION_NAME = "pitchfork_reviews"


def chunk_text(text: str, chunk_size: int = 4, overlap: int = 1) -> list[str]:
    """Splits text into overlapping chunks of sentences."""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if not sentences:
        return []
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = ". ".join(sentences[i:i + chunk_size])
        if chunk:
            chunks.append(chunk + '.')
    return chunks


async def generate_tags_for_chunk(client: httpx.AsyncClient, chunk: str, token: str) -> list[str]:
    """Uses Gemini to extract descriptive tags from a text chunk, with retries."""
    prompt = (
        "You are an expert musicologist. Analyze the following excerpt from a music review. "
        "Extract a list of 5-7 descriptive keywords and phrases that capture the mood, genre, "
        "instrumentation, and overall sonic texture. Focus on evocative adjectives.\n\n"
        f"REVIEW EXCERPT:\n\"...{chunk}...\"\n\n"
        "Return ONLY a JSON list of strings. For example: "
        "[\"dream-pop\", \"shimmering guitars\", \"hazy atmosphere\", \"ethereal vocals\", \"introspective\"]"
    )

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    for attempt in range(5):
        try:
            response = await client.post(api_url, json=payload, headers=headers, timeout=45.0)
            response.raise_for_status()
            response_data = response.json()
            content = response_data['candidates'][0]['content']['parts'][0]['text']
            json_str = content.strip().replace("```json", "").replace("```", "")
            return json.loads(json_str)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [429, 503]:
                wait_time = (2 ** attempt) + 1
                print(f"API Error {e.response.status_code}. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                print(f"Non-retryable HTTP Error generating tags: {e.response.text}")
                return []
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return []

    print("API is still unavailable after multiple retries. Skipping chunk.")
    return []


async def process_review_batch(rows: pd.DataFrame, collection: chromadb.Collection, model: SentenceTransformer, token: str):
    """Processes a batch of reviews: chunks, tags, embeds, and saves to DB."""
    enriched_chunks = []
    async with httpx.AsyncClient() as client:
        for _, row in rows.iterrows():
            text_chunks = chunk_text(row['review_text'])
            tasks = [generate_tags_for_chunk(client, chunk, token) for chunk in text_chunks]
            tags_results = await asyncio.gather(*tasks)

            for chunk, tags in zip(text_chunks, tags_results):
                if tags:
                    enriched_chunks.append({
                        "artist": row['artist'], "album_title": row['album_title'],
                        "score": row['score'], "review_url": row['review_url'],
                        "text_chunk": chunk, "tags": tags
                    })

    if not enriched_chunks:
        return 0

    enriched_df = pd.DataFrame(enriched_chunks)
    enriched_df['search_document'] = enriched_df.apply(
        lambda row: f"Tags: {', '.join(row['tags'])}. Review excerpt: {row['text_chunk']}", axis=1
    )

    embeddings = model.encode(enriched_df['search_document'].tolist())

    current_count = collection.count()
    ids = [f"chunk_{current_count + i}" for i in range(len(enriched_df))]
    documents = enriched_df['search_document'].tolist()

    metadatas = []
    for i, row in enriched_df.iterrows():
        meta = row.to_dict()
        meta.pop('search_document', None)
        meta['tags'] = json.dumps(meta['tags'])
        metadatas.append(meta)

    collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas)
    return len(enriched_df)


async def main():
    """Main function to build and load the knowledge base in resilient batches."""
    logging.getLogger('chromadb').setLevel(logging.WARNING)

    print("Getting GCP Authentication Token...")
    auth_token = get_gcp_auth_token()
    if not auth_token: return
    token_generation_time = time.time()
    TOKEN_LIFESPAN_SECONDS = 45 * 60

    try:
        raw_df = pd.read_json(RAW_DATA_FILE, lines=True)
    except FileNotFoundError:
        print(f"FATAL: Raw data file not found at '{RAW_DATA_FILE}'.")
        return
    df = raw_df[raw_df['artist'] != 'N/A'].copy()

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    existing_items = collection.get(include=["metadatas"])
    processed_urls = {item['review_url'] for item in existing_items['metadatas']} if existing_items['metadatas'] else set()
    print(f"Found {len(processed_urls)} already processed reviews. They will be skipped.")

    unprocessed_df = df[~df['review_url'].isin(processed_urls)]
    if unprocessed_df.empty:
        print("All reviews have already been processed. Nothing to do.")
        return
    print(f"Found {len(unprocessed_df)} new reviews to process.")

    # --- THE FIX: Process and save data in smaller batches ---
    batch_size = 5 # Process 5 reviews at a time
    model = SentenceTransformer(MODEL_NAME)

    with tqdm(total=len(unprocessed_df), desc="Processing reviews") as pbar:
        for i in range(0, len(unprocessed_df), batch_size):
            # Refresh token if needed
            if time.time() - token_generation_time > TOKEN_LIFESPAN_SECONDS:
                print("\nRefreshing auth token...")
                auth_token = get_gcp_auth_token()
                token_generation_time = time.time()
                print("Token refreshed.")

            batch_df = unprocessed_df.iloc[i:i + batch_size]
            await process_review_batch(batch_df, collection, model, auth_token)
            pbar.update(len(batch_df))

    count = collection.count()
    print(f"\nKnowledge base update complete. Collection now contains {count} items.")


if __name__ == "__main__":
    asyncio.run(main())
