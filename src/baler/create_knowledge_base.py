import asyncio
import logging
import httpx
import pandas as pd
from tqdm import tqdm

from . import config
from .database import VectorDB
from .llm import GeminiClient
from .utils import chunk_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def main():
    """Main function to orchestrate the knowledge base build."""
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        db = VectorDB()
        llm = GeminiClient()
        raw_df = pd.read_json(config.RAW_DATA_FILE, lines=True)
    except FileNotFoundError:
        logging.fatal(f"FATAL: Raw data file not found at '{config.RAW_DATA_FILE}'.")
        return
    except Exception as e:
        logging.fatal(f"Failed to initialize services: {e}")
        return

    logging.info("Starting regular knowledge base build process...")
    
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
