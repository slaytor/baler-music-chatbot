import httpx
import json
import pandas as pd
from tqdm import tqdm
import logging
import time
import asyncio

# --- REFACTORED IMPORTS ---
from . import config
from .database import VectorDB
from .llm import GeminiClient
from .utils import chunk_text

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def main():
    """Main function to orchestrate the knowledge base build."""
    logging.getLogger('chromadb').setLevel(logging.WARNING)

    # 1. Initialize services
    try:
        db = VectorDB()
        llm = GeminiClient()
    except Exception as e:
        logging.fatal(f"Failed to initialize services: {e}")
        return

    # 2. Load and filter data
    try:
        raw_df = pd.read_json(config.RAW_DATA_FILE, lines=True)
    except FileNotFoundError:
        logging.fatal(f"FATAL: Raw data file not found at '{config.RAW_DATA_FILE}'.")
        return

    original_count = len(raw_df)
    logging.info(f"Loaded {original_count} total records from {config.RAW_DATA_FILE}.")

    raw_df.drop_duplicates(subset=['review_url'], keep='last', inplace=True)
    logging.info(f"Found {len(raw_df)} unique reviews after deduplication.")

    df = raw_df[raw_df['artist'] != 'N/A'].copy()
    logging.info(f"Filtered down to {len(df)} valid, unique records.")

    # 3. Find unprocessed reviews
    processed_urls = db.get_processed_urls()
    logging.info(f"Found {len(processed_urls)} already processed unique reviews. They will be skipped.")

    unprocessed_df = df[~df['review_url'].isin(processed_urls)]
    if unprocessed_df.empty:
        logging.info("All reviews have already been processed. Nothing to do.")
        return
    logging.info(f"Found {len(unprocessed_df)} new reviews to process.")

    # 4. Process and save in batches
    with tqdm(total=len(unprocessed_df), desc="Processing reviews") as pbar:
        for i in range(0, len(unprocessed_df), config.KB_BATCH_SIZE):
            batch_df = unprocessed_df.iloc[i:i + config.KB_BATCH_SIZE]

            enriched_chunks = []
            async with httpx.AsyncClient() as client:
                for _, row in batch_df.iterrows():
                    text_chunks = chunk_text(row['review_text'])

                    # Generate tags for all chunks of a review concurrently
                    tasks = [llm.generate_tags_for_chunk(client, chunk) for chunk in text_chunks]
                    tags_results = await asyncio.gather(*tasks)

                    for chunk, tags in zip(text_chunks, tags_results):
                        if tags:
                            enriched_chunks.append({
                                "artist": row['artist'], "album_title": row['album_title'],
                                "score": row['score'], "review_url": row['review_url'],
                                "text_chunk": chunk, "tags": tags
                            })

            if not enriched_chunks:
                pbar.update(len(batch_df))
                continue

            # 5. Add batch to database
            enriched_df = pd.DataFrame(enriched_chunks)
            db.add_batch(enriched_df) # Embedding is now handled inside db.add_batch

            pbar.update(len(batch_df))

            # Add inter-batch delay
            await asyncio.sleep(config.INTER_BATCH_DELAY_SECONDS)

    count = db.get_count()
    logging.info(f"\nKnowledge base update complete. Collection now contains {count} items.")


if __name__ == "__main__":
    asyncio.run(main())
