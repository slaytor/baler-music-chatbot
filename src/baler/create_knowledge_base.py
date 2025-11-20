import asyncio
import logging
import os
import httpx
import pandas as pd
from tqdm import tqdm

from . import config
from .database import VectorDB
from .llm import GeminiClient
from .utils import chunk_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- MIGRATION CONFIG ---
MIGRATION_FILE = config.PROJECT_ROOT / "enriched_data.jsonl"

async def main():
    """Main function to orchestrate the knowledge base build."""
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        db = VectorDB()
    except Exception as e:
        logging.fatal(f"Failed to initialize database service: {e}")
        return

    # --- MIGRATION PATH ---
    if MIGRATION_FILE.exists():
        logging.warning("*" * 60)
        logging.warning("!!! MIGRATION MODE DETECTED !!!")
        logging.warning(f"Found migration file: {MIGRATION_FILE}")
        logging.warning("This script will now import data from this file WITHOUT making any new API calls.")
        logging.warning("*" * 60)

        try:
            enriched_df = pd.read_json(MIGRATION_FILE, lines=True)
            logging.info(f"Loaded {len(enriched_df)} records from migration file.")
            
            db.add_batch(enriched_df)
            
            logging.info("--- MIGRATION COMPLETE ---")
            logging.warning("IMPORTANT: The migration was successful. Please manually delete the 'enriched_data.jsonl' file now.")
            logging.info(f"Database now contains {db.get_count()} items.")
            return
        except Exception as e:
            logging.fatal(f"FATAL: An error occurred during migration: {e}")
            return

    # --- REGULAR EXECUTION PATH ---
    logging.info("Starting regular knowledge base build process...")
    try:
        llm = GeminiClient()
        raw_df = pd.read_json(config.RAW_DATA_FILE, lines=True)
    except FileNotFoundError:
        logging.fatal(f"FATAL: Raw data file not found at '{config.RAW_DATA_FILE}'.")
        return
    except Exception as e:
        logging.fatal(f"Failed to initialize services: {e}")
        return

    original_count = len(raw_df)
    logging.info(f"Loaded {original_count} total records from {config.RAW_DATA_FILE}.")

    raw_df.drop_duplicates(subset=["review_url"], keep="last", inplace=True)
    df = raw_df[raw_df["artist"] != "N/A"].copy()
    logging.info(f"Filtered down to {len(df)} valid, unique records.")

    processed_urls = db.get_processed_urls()
    logging.info(f"Found {len(processed_urls)} already processed reviews to skip.")

    unprocessed_df = df[~df["review_url"].isin(processed_urls)]
    if unprocessed_df.empty:
        logging.info("All reviews have already been processed. Nothing to do.")
        return
    logging.info(f"Found {len(unprocessed_df)} new reviews to process.")

    with tqdm(total=len(unprocessed_df), desc="Processing reviews") as pbar:
        for i in range(0, len(unprocessed_df), config.KB_BATCH_SIZE):
            batch_df = unprocessed_df.iloc[i : i + config.KB_BATCH_SIZE]
            enriched_chunks = []
            async with httpx.AsyncClient() as client:
                for _, row in batch_df.iterrows():
                    text_chunks = chunk_text(row["review_text"])
                    tasks = [llm.generate_tags_for_chunk(client, chunk) for chunk in text_chunks]
                    tags_results = await asyncio.gather(*tasks)
                    for chunk, tags in zip(text_chunks, tags_results):
                        if tags:
                            enriched_chunks.append({
                                "artist": row["artist"], "album_title": row["album_title"],
                                "score": row["score"], "review_url": row["review_url"],
                                "text_chunk": chunk, "tags": tags
                            })
            if enriched_chunks:
                enriched_batch_df = pd.DataFrame(enriched_chunks)
                db.add_batch(enriched_batch_df)
            pbar.update(len(batch_df))
            await asyncio.sleep(config.INTER_BATCH_DELAY_SECONDS)

    logging.info(f"\nKnowledge base update complete. Collection now contains {db.get_count()} items.")

if __name__ == "__main__":
    asyncio.run(main())
