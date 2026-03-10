"""
Downloads albums.json from the-fork.vercel.app, compares against the local
reviews.jsonl (and reviews_new.jsonl if present), and writes the missing
Pitchfork review URLs to a file.

Usage:
    poetry run python -m src.baler.build_scrape_list

Output:
    scrape_list.txt  — one URL per line, ready for the spider's url_file param
"""

import json
import logging

import httpx

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ALBUMS_JSON_URL = "https://the-fork.vercel.app/albums.json"
OUTPUT_FILE = config.PROJECT_ROOT / "scrape_list.txt"
PITCHFORK_BASE = "https://pitchfork.com"

REVIEW_FILES = [
    config.PROJECT_ROOT / "reviews.jsonl",
    config.PROJECT_ROOT / "reviews_new.jsonl",
]


def load_urls_from_file(path) -> set[str]:
    if not path.exists():
        return set()
    urls = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                review = json.loads(line)
                if url := review.get("review_url"):
                    urls.add(url)
            except json.JSONDecodeError:
                pass
    logging.info(f"Loaded {len(urls):,} existing URLs from {path.name}")
    return urls


def load_existing_urls() -> set[str]:
    all_urls: set[str] = set()
    for path in REVIEW_FILES:
        all_urls |= load_urls_from_file(path)
    logging.info(f"Total existing URLs across all files: {len(all_urls):,}")
    return all_urls


def main():
    logging.info(f"Fetching {ALBUMS_JSON_URL}...")
    r = httpx.get(ALBUMS_JSON_URL, timeout=60, follow_redirects=True)
    r.raise_for_status()
    albums = r.json()
    logging.info(f"Downloaded {len(albums):,} albums from the-fork")

    existing_urls = load_existing_urls()

    new_urls = []
    for album in albums:
        url_path = album.get("url", "")
        if not url_path:
            continue
        full_url = PITCHFORK_BASE + url_path.rstrip("/") + "/"
        if full_url not in existing_urls:
            new_urls.append(full_url)

    logging.info(f"New albums to scrape: {len(new_urls):,} (already have {len(existing_urls):,})")

    OUTPUT_FILE.write_text("\n".join(new_urls))
    logging.info(f"Written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
