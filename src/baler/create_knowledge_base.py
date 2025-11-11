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
CONCURRENT_REQUESTS = 5
INTER_BATCH_DELAY_SECONDS = 1 # Add a 1-second pause between batches


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

    # Increase max attempts slightly for more resilience
    for attempt in range(7):
        try:
            response = await client.post(api_url, json=payload, headers=headers, timeout=45.0)
            response.raise_for_status()
            response_data = response.json()
            if 'candidates' in response_data and response_data['candidates'] and \
               'content' in response_data['candidates'][0] and \
               'parts' in response_data['candidates'][0]['content'] and \
               response_data['candidates'][0]['content']['parts']:
                content = response_data['candidates'][0]['content']['parts'][0]['text']
                json_str = content.strip().replace("```json", "").replace("```", "")
                if not json_str:
                    print(f"Warning: Received empty content from API for chunk. Skipping.")
                    return []
                return json.loads(json_str)
            else:
                print(f"Warning: Unexpected API response structure: {response_data}")
                return []
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [429, 503]:
                # Increase base wait time and exponent for more aggressive backoff
                wait_time = (2 ** (attempt + 1)) + 1 # 3, 5, 9, 17, 33... seconds
                print(f"API Error {e.response.status_code}. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                print(f"Non-retryable HTTP Error generating tags: {e.response.text}")
                return []
        except json.JSONDecodeError as e:
             content_for_error = "N/A"
             if 'content' in locals(): content_for_error = content
             print(f"Error decoding JSON response: {e}. Content: '{content_for_error}'")
             return []
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return []

    print("API is still unavailable after multiple retries. Skipping chunk.")
    return []


async def main():
    """Main function to build and load the knowledge base."""
    logging.getLogger('chromadb').setLevel(logging.WARNING)

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
    print(f"Found {len(processed_urls)} already processed unique reviews. They will be skipped.")

    unprocessed_df = df[~df['review_url'].isin(processed_urls)]
    if unprocessed_df.empty:
        print("All reviews have already been processed. Nothing to do."); return
    print(f"Found {len(unprocessed_df)} new reviews to process.")

    all_chunks_to_process = []
    for _, row in unprocessed_df.iterrows():
        text_chunks = chunk_text(row['review_text'])
        for chunk in text_chunks:
            if chunk and chunk.strip() != '.':
                 all_chunks_to_process.append({"row_data": row.to_dict(), "chunk": chunk})

    print(f"Total new chunks to process: {len(all_chunks_to_process)}")

    enriched_chunks_data = []
    tasks = []

    async with httpx.AsyncClient() as http_client:
        pbar = tqdm(total=len(all_chunks_to_process), desc="Generating tags concurrently")
        for i, item in enumerate(all_chunks_to_process):
            if time.time() - token_generation_time > TOKEN_LIFESPAN_SECONDS:
                print("\nRefreshing auth token...")
                auth_token = get_gcp_auth_token()
                token_generation_time = time.time()
                if not auth_token: print("FATAL: Failed to refresh token."); break
                print("Token refreshed.")

            task = generate_tags_for_chunk(http_client, item['chunk'], auth_token)
            tasks.append((item, task))

            if len(tasks) >= CONCURRENT_REQUESTS or i == len(all_chunks_to_process) - 1:
                results = await asyncio.gather(*(t for _, t in tasks))

                for (original_item, _), tags in zip(tasks, results):
                    if tags:
                        enriched_chunks_data.append({
                            **original_item['row_data'],
                            "text_chunk": original_item['chunk'],
                            "tags": tags
                        })
                pbar.update(len(tasks))
                tasks = []
                # --- THE FIX: Add a small delay between batches ---
                await asyncio.sleep(INTER_BATCH_DELAY_SECONDS)
        pbar.close()

    if not enriched_chunks_data:
        print("No new chunks were generated during this run. Exiting."); return

    print(f"\nGenerated tags for {len(enriched_chunks_data)} new chunks. Now saving progress...")
    enriched_df = pd.DataFrame(enriched_chunks_data)

    required_cols = ['artist', 'album_title', 'score', 'review_url', 'text_chunk', 'tags']
    if not all(col in enriched_df.columns for col in required_cols):
        print("Error: DataFrame missing required columns."); print(enriched_df.head()); return

    enriched_df['search_document'] = enriched_df.apply(
        lambda row: f"Tags: {', '.join(row['tags'])}. Review excerpt: {row['text_chunk']}", axis=1
    )

    print("Embedding search documents...")
    model = SentenceTransformer(MODEL_NAME)
    if enriched_df.empty: print("No documents to embed.")
    else: embeddings = model.encode(enriched_df['search_document'].tolist(), show_progress_bar=True)

    if not enriched_df.empty:
        current_count = collection.count()
        ids = [f"chunk_{current_count + i}" for i in range(len(enriched_df))]
        documents = enriched_df['search_document'].tolist()

        metadatas = []
        for _, row in enriched_df.iterrows():
            meta = row.to_dict(); meta.pop('search_document', None)
            meta['tags'] = json.dumps(meta.get('tags', []))
            for key in ['artist', 'album_title', 'score', 'review_url', 'text_chunk']:
                if key not in meta: meta[key] = "N/A"
            metadatas.append(meta)

        batch_size = 100
        print(f"Saving {len(enriched_df)} new documents to ChromaDB...")
        for i in tqdm(range(0, len(enriched_df), batch_size), desc="Saving to DB"):
            try:
                collection.add(
                    ids=ids[i:i+batch_size],
                    embeddings=embeddings[i:i+batch_size].tolist(),
                    metadatas=metadatas[i:i+batch_size]
                )
            except Exception as e:
                print(f"Error adding batch to ChromaDB: {e}"); break

    count = collection.count()
    print(f"Knowledge base update complete. Collection now contains {count} items.")


if __name__ == "__main__":
    asyncio.run(main())
