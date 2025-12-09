import asyncio
import logging
import httpx
import pandas as pd
import argparse
import json
from tqdm import tqdm

from . import config
from .database import VectorDB
from .llm import get_llm_client
from .utils import chunk_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def process_chunk_with_semaphore(semaphore, llm, client, chunk, row_data):
    async with semaphore:
        tags = await llm.generate_tags_for_chunk(client, chunk)
        if tags:
            return {
                "artist": row_data["artist"], "album_title": row_data["album_title"],
                "score": row_data["score"], "review_url": row_data["review_url"],
                "text_chunk": chunk, "tags": tags,
                "album_cover_url": row_data.get("album_cover_url", "N/A")
            }
        return None

def load_reviews_robustly(file_path):
    """Reads a JSONL file line by line, skipping any malformed lines."""
    records = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed JSON on line {i+1} in {file_path}")
    return pd.DataFrame(records)

async def main():
    """Main function to orchestrate the knowledge base build."""
    parser = argparse.ArgumentParser(description="Build the knowledge base from a JSONL file.")
    parser.add_argument(
        "--input-file",
        type=str,
        default=config.RAW_DATA_FILE,
        help="Path to the input JSONL file of reviews."
    )
    args = parser.parse_args()

    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        db = VectorDB()
        llm = get_llm_client()
        logging.info(f"Loading reviews from {args.input_file}...")
        raw_df = load_reviews_robustly(args.input_file)
    except FileNotFoundError:
        logging.fatal(f"FATAL: Raw data file not found at '{args.input_file}'.")
        return
    except Exception as e:
        logging.fatal(f"Failed to initialize services: {e}")
        return

    logging.info(f"Starting knowledge base build process using {config.LLM_PROVIDER}...")
    
    original_count = len(raw_df)
    logging.info(f"Loaded {original_count} total records.")

    raw_df.drop_duplicates(subset=["review_url"], keep="last", inplace=True)
    df = raw_df[raw_df["artist"] != "N/A"].copy()
    logging.info(f"Filtered down to {len(df)} valid, unique records.")

    processed_urls = db.get_processed_urls()
    logging.info(f"Found {len(processed_urls)} already processed reviews to skip.")

    unprocessed_df = df[~df["review_url"].isin(processed_urls)]
    if unprocessed_df.empty:
        logging.info("All reviews in the input file have already been processed. Nothing to do.")
        return
    logging.info(f"Found {len(unprocessed_df)} new reviews to process.")

    CONCURRENCY_LIMIT = 5
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    with tqdm(total=len(unprocessed_df), desc="Processing reviews") as pbar:
        for i in range(0, len(unprocessed_df), config.KB_BATCH_SIZE):
            batch_df = unprocessed_df.iloc[i : i + config.KB_BATCH_SIZE]
            
            async with httpx.AsyncClient() as client:
                tasks = []
                for _, row in batch_df.iterrows():
                    text_chunks = chunk_text(row["review_text"])
                    row_data = row.to_dict()
                    for chunk in text_chunks:
                        tasks.append(process_chunk_with_semaphore(semaphore, llm, client, chunk, row_data))
                
                results = await asyncio.gather(*tasks)
                enriched_chunks = [res for res in results if res is not None]

            if enriched_chunks:
                enriched_batch_df = pd.DataFrame(enriched_chunks)
                db.add_batch(enriched_batch_df)
            pbar.update(len(batch_df))
            await asyncio.sleep(config.INTER_BATCH_DELAY_SECONDS)

    logging.info(f"\nKnowledge base update complete. Collection now contains {db.get_count()} items.")

if __name__ == "__main__":
    asyncio.run(main())
