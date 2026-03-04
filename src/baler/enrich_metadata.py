"""
One-time migration script: enriches ChromaDB records with Spotify metadata.

For each unique album in the collection, fetches:
- artist_genres  (list of genre strings)
- label          (record label string)
- related_artists (list of related artist names)

Updates all chunks belonging to that album with:
- new metadata fields (artist_genres, label, related_artists as JSON strings)
- a richer search_document string that includes the above for better BM25 matching

Run once from the project root:
    poetry run python -m src.baler.enrich_metadata

Rate limiting: ~200ms sleep between albums → ~15–20 min for ~4,500 albums.
"""

import asyncio
import json
import logging

import chromadb

from . import config
from .music_services import SpotifyClient, SpotifyRateLimitError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


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


def fetch_all_records(collection) -> tuple[list, list, list]:
    """Fetch all IDs, metadatas, and documents from the collection in batches."""
    count = collection.count()
    logging.info(f"Total records in collection: {count}")

    all_ids, all_metadatas, all_documents = [], [], []
    batch_size = 250

    for offset in range(0, count, batch_size):
        logging.info(f"Fetching batch at offset {offset}/{count}...")
        items = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas", "documents"],
        )
        if items and items.get("ids"):
            all_ids.extend(items["ids"])
            all_metadatas.extend(items["metadatas"])
            all_documents.extend(items["documents"])

    return all_ids, all_metadatas, all_documents


def build_unique_albums(ids: list, metadatas: list, documents: list) -> dict[str, list]:
    """Group chunk indices by review_url."""
    albums: dict[str, list] = {}
    for i, meta in enumerate(metadatas):
        url = meta.get("review_url")
        if url:
            albums.setdefault(url, []).append(i)
    return albums


def build_search_document(meta: dict, spotify_data: dict) -> str:
    tags = (
        json.loads(meta.get("tags", "[]"))
        if isinstance(meta.get("tags"), str)
        else meta.get("tags", [])
    )
    genres = spotify_data.get("artist_genres", [])
    related = spotify_data.get("related_artists", [])
    label = spotify_data.get("label", "N/A")
    return (
        f"Artist: {meta.get('artist', '')}. "
        f"Genres: {', '.join(genres)}. "
        f"Label: {label}. "
        f"Related: {', '.join(related)}. "
        f"Tags: {', '.join(tags)}. "
        f"Review: {meta.get('text_chunk', '')}"
    )


async def run_enrichment():
    collection = get_chroma_collection()
    spotify = SpotifyClient()

    logging.info("Fetching all records from ChromaDB...")
    all_ids, all_metadatas, all_documents = fetch_all_records(collection)

    albums = build_unique_albums(all_ids, all_metadatas, all_documents)
    logging.info(f"Found {len(albums)} unique albums to enrich.")

    enriched = 0
    skipped = 0
    already_done = 0

    for album_num, (review_url, indices) in enumerate(albums.items(), start=1):
        sample_meta = all_metadatas[indices[0]]
        album_title = sample_meta.get("album_title", "")
        artist = sample_meta.get("artist", "")

        # Skip albums already enriched in a previous run
        if sample_meta.get("artist_genres") not in (None, "", "N/A", "[]"):
            already_done += 1
            if already_done % 100 == 0:
                logging.info(
                    f"[{album_num}/{len(albums)}] Skipping already-enriched albums... ({already_done} so far)"
                )
            continue

        logging.info(f"[{album_num}/{len(albums)}] {artist} — {album_title}")

        try:
            spotify_data = await spotify.get_album_metadata(album_title, artist)
        except SpotifyRateLimitError as e:
            logging.warning(f"  Spotify rate limit hit. Retry-After: {e.retry_after}s")
            if e.retry_after > 300:
                logging.error(
                    f"  Retry-After is {e.retry_after}s — this looks like a daily quota reset, not a per-minute rate limit. "
                    f"Aborting. Try again in {e.retry_after // 3600:.1f} hours."
                )
                break
            logging.info(f"  Waiting {e.retry_after}s then retrying...")
            await asyncio.sleep(e.retry_after)
            try:
                spotify_data = await spotify.get_album_metadata(album_title, artist)
            except SpotifyRateLimitError as e2:
                logging.error(
                    f"  Still rate limited after waiting ({e2.retry_after}s). Aborting."
                )
                break

        if spotify_data is None:
            logging.warning("  Spotify: not found, skipping.")
            skipped += 1
            await asyncio.sleep(2.0)
            continue

        logging.info(
            f"  genres={spotify_data['artist_genres'][:3]}, "
            f"label={spotify_data['label']}, "
            f"related={spotify_data['related_artists'][:3]}"
        )

        # Build updated IDs, metadatas, documents for all chunks of this album
        chunk_ids = [all_ids[i] for i in indices]
        updated_metadatas = []
        updated_documents = []

        for i in indices:
            meta = dict(all_metadatas[i])
            meta["artist_genres"] = json.dumps(spotify_data["artist_genres"])
            meta["label"] = spotify_data["label"] or meta.get("label", "N/A")
            meta["related_artists"] = json.dumps(spotify_data["related_artists"])
            updated_metadatas.append(meta)
            updated_documents.append(build_search_document(meta, spotify_data))

        try:
            # Update metadata and stored document (no re-embedding needed)
            collection.update(
                ids=chunk_ids,
                metadatas=updated_metadatas,
                documents=updated_documents,
            )
            enriched += 1
        except Exception as e:
            logging.error(f"  Failed to update chunks: {e}")

        # Rate-limit: 2s between albums to avoid Spotify 429s
        await asyncio.sleep(2.0)

    logging.info(
        f"Enrichment complete. Enriched: {enriched}, "
        f"Already done: {already_done}, Skipped (not on Spotify): {skipped}."
    )


if __name__ == "__main__":
    asyncio.run(run_enrichment())
