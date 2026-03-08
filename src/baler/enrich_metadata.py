"""
Enrichment script: updates ChromaDB records with Last.fm artist metadata.

For each unique album in the collection, fetches from Last.fm:
- artist_genres  (top user-applied tags)
- related_artists (similar artists by listening patterns)

Existing label data is preserved as-is. artist_genres and related_artists
are overwritten for every album regardless of prior enrichment.

Run from the project root:
    poetry run python -m src.baler.enrich_metadata

Processes up to CONCURRENCY albums in parallel. Resumable: safe to kill and re-run.
"""

import asyncio
import json
import logging

import chromadb

from . import config
from .music_services import LastFmClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BATCH_SIZE = 250
CONCURRENCY = 5


def get_chroma_collection():
    if config.DB_PROVIDER.upper() == "CLOUD":
        client = chromadb.CloudClient(
            api_key=config.CHROMA_CLOUD_API_KEY,
            tenant=config.CHROMA_CLOUD_TENANT,
            database=config.CHROMA_CLOUD_DATABASE,
        )
    else:
        client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
    return client.get_or_create_collection(name=config.COLLECTION_NAME)


def build_search_document(meta: dict, lastfm_data: dict) -> str:
    tags = (
        json.loads(meta.get("tags", "[]"))
        if isinstance(meta.get("tags"), str)
        else meta.get("tags", [])
    )
    genres = lastfm_data.get("artist_genres", [])
    related = lastfm_data.get("related_artists", [])
    label = meta.get("label", "N/A")
    return (
        f"Artist: {meta.get('artist', '')}. "
        f"Genres: {', '.join(genres)}. "
        f"Label: {label}. "
        f"Related: {', '.join(related)}. "
        f"Tags: {', '.join(tags)}. "
        f"Review: {meta.get('text_chunk', '')}"
    )


async def enrich_album(collection, lastfm: LastFmClient, review_url: str, album_num: int) -> bool:
    """Fetch Last.fm metadata and overwrite artist_genres + related_artists for all chunks of one album."""
    loop = asyncio.get_event_loop()

    result = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: collection.get(
                where={"review_url": review_url},
                include=["metadatas", "documents"],
            ),
        ),
        timeout=60,
    )

    if not result or not result.get("ids"):
        return False

    sample_meta = result["metadatas"][0]
    artist = sample_meta.get("artist", "")
    album_title = sample_meta.get("album_title", "")

    logging.info(f"[{album_num}] {artist} — {album_title}")

    lastfm_data = await lastfm.get_metadata(artist, album_title)

    if not lastfm_data["artist_genres"] and not lastfm_data["related_artists"]:
        logging.warning(f"  Last.fm: no data found for '{artist}'")

    logging.info(
        f"  genres={lastfm_data['artist_genres'][:3]}, "
        f"related={lastfm_data['related_artists'][:3]}"
    )

    chunk_ids = result["ids"]
    updated_metadatas = []
    updated_documents = []

    for meta in result["metadatas"]:
        meta = dict(meta)
        meta["artist_genres"] = json.dumps(lastfm_data["artist_genres"])
        meta["related_artists"] = json.dumps(lastfm_data["related_artists"])
        updated_metadatas.append(meta)
        updated_documents.append(build_search_document(meta, lastfm_data))

    try:
        await loop.run_in_executor(
            None,
            lambda: collection.update(
                ids=chunk_ids,
                metadatas=updated_metadatas,
                documents=updated_documents,
            ),
        )
    except Exception as e:
        logging.error(f"  Failed to update chunks: {e}")
        return False

    return True


async def run_enrichment():
    collection = get_chroma_collection()
    lastfm = LastFmClient()
    loop = asyncio.get_event_loop()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    count = collection.count()
    logging.info(f"Total records in collection: {count}")

    processed_urls: set[str] = set()
    enriched = 0
    skipped = 0
    album_num = 0

    async def bounded_enrich(url: str, num: int) -> bool:
        async with semaphore:
            try:
                return await enrich_album(collection, lastfm, url, num)
            except TimeoutError:
                logging.warning(f"  [{num}] Timeout, skipping.")
                return False
            except Exception as e:
                logging.error(f"  [{num}] Unexpected error: {e}")
                return False

    for offset in range(0, count, BATCH_SIZE):
        logging.info(f"Scanning batch at offset {offset}/{count}...")

        try:
            items = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda o=offset: collection.get(
                        limit=BATCH_SIZE,
                        offset=o,
                        include=["metadatas"],
                    ),
                ),
                timeout=60,
            )
        except TimeoutError:
            logging.warning(f"  Timeout fetching batch at offset {offset}, skipping.")
            continue

        if not items or not items.get("ids"):
            continue

        # Collect new URLs from this batch
        batch_tasks = []
        for meta in items["metadatas"]:
            url = meta.get("review_url")
            if not url or url in processed_urls:
                continue
            processed_urls.add(url)
            album_num += 1
            batch_tasks.append(bounded_enrich(url, album_num))

        # Enrich all new albums in this batch concurrently
        results = await asyncio.gather(*batch_tasks)
        enriched += sum(1 for r in results if r)
        skipped += sum(1 for r in results if not r)

    logging.info(
        f"Enrichment complete. Enriched: {enriched}, Skipped: {skipped}."
    )


if __name__ == "__main__":
    asyncio.run(run_enrichment())
