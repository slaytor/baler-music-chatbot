# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Get 'Forked** (codenamed *Baler*) is an AI-powered music recommendation app (live at [get-forked.org](https://get-forked.org)) that uses RAG over Pitchfork album reviews to suggest albums based on natural language queries. It uses ChromaDB for vector search, Gemini for LLM inference, SentenceTransformers for embeddings, and Spotify for album links.

## Workflow Preferences

- **Always ask before making changes** — propose and confirm before editing files
- **Move fast** — solo project, large feature branches are fine, prioritize working code
- **Ruff** for linting/formatting (replaces black + isort)
- **Never auto-commit** — always let Sam review diffs before committing
- **Always preserve the NDJSON streaming contract** in `/recommend` — the frontend depends on the exact `{chunk}`, `{sources}`, `{remaining_sources}`, `{error}` format
- **Never modify scraper CSS selectors** without flagging it — Pitchfork markup is fragile and changes break the whole pipeline

## Commands

All commands use Poetry. Run from the project root.

```bash
# Install dependencies
poetry install

# Run the web app locally (requires .env)
poetry run uvicorn src.baler.main:app --host 0.0.0.0 --port 8080 --reload

# Run with Docker (recommended for full stack)
docker-compose up --build

# Run the Scrapy spider manually (from src/ directory)
cd src && poetry run scrapy crawl pitchfork_reviews -o ../reviews_today.jsonl

# Run spider with deduplication against a previous file
cd src && poetry run scrapy crawl pitchfork_reviews -a previous_file=../reviews.jsonl -o ../reviews_today.jsonl

# Sync new scraped data from S3 to local reviews.jsonl
poetry run python -m src.baler.update_raw_data

# Lint and format
poetry run ruff check src/
poetry run ruff format src/

# Build / update the vector knowledge base from reviews.jsonl
poetry run python -m src.baler.create_knowledge_base

# Build KB from a specific input file
poetry run python -m src.baler.create_knowledge_base --input-file reviews_today.jsonl

# Check database item count
poetry run python -m src.baler.check_db

# Check DB and sample N items
poetry run python -m src.baler.check_db 5
```

## Architecture

### Data Flow

```
Pitchfork.com
    → Scrapy/Playwright spider (src/baler/spiders/scraper.py)
    → Daily JSONL uploaded to AWS S3 (daily_scrapes/)
    → update_raw_data.py syncs S3 → local reviews.jsonl
    → create_knowledge_base.py: chunks text → LLM generates tags → embeds → upserts into ChromaDB
    → FastAPI app queries ChromaDB → streams Gemini response to frontend
```

### Key Modules

- **`config.py`** — All configuration and env vars. Single source of truth for paths, model names, credentials, and provider switches.
- **`main.py`** — FastAPI app. Two endpoints: `POST /recommend` (main RAG stream) and `POST /find-album-url` (Spotify lookup). Serves `static/index.html` at root.
- **`database.py`** — `VectorDB` class wrapping ChromaDB. Handles both Cloud and local HTTP client modes. Uses `all-MiniLM-L6-v2` for embeddings. The `search()` method manually implements pagination since ChromaDB's `query()` has no native offset.
- **`llm.py`** — Factory pattern (`get_llm_client()`) returning either `GeminiClient` or `OllamaClient`. Both implement `async stream_response()` yielding NDJSON chunks. GeminiClient uses `google-auth` for auto-refreshing credentials from `gcloud-credentials.json`.
- **`create_knowledge_base.py`** — Ingestion pipeline. Reads JSONL, chunks review text, calls LLM to generate semantic tags per chunk, then batches into ChromaDB. Skips already-processed URLs.
- **`music_services.py`** — `SpotifyClient` using Client Credentials flow for album URL lookups.
- **`spiders/scraper.py`** — Scrapy+Playwright spider crawling Pitchfork. Stops pagination when it encounters a previously-seen URL.
- **`update_raw_data.py`** — Downloads new daily JSONL files from S3, deduplicates by `review_url`, appends to master `reviews.jsonl`.

### LLM Providers

Configured via env vars with two separate settings:
- `LLM_PROVIDER` — used during KB ingestion (`create_knowledge_base.py`). Defaults to `OLLAMA` locally.
- `APP_LLM_PROVIDER` — used by the live web app. Defaults to `GEMINI`.

Switching to Ollama locally requires a running Ollama instance with the `mistral` model.

### Database Modes

`DB_PROVIDER` env var controls ChromaDB connection:
- `CLOUD` (default) — connects to Chroma Cloud using `CHROMA_CLOUD_API_KEY`, `CHROMA_CLOUD_TENANT`, `CHROMA_CLOUD_DATABASE`
- `LOCAL` — connects to a local ChromaDB HTTP server at `CHROMA_HOST:CHROMA_PORT`

### Streaming Response Format

The `/recommend` endpoint returns NDJSON. Each line is a JSON object with one of:
- `{"chunk": "..."}` — streaming LLM text
- `{"sources": [...]}` — the top-k albums used as context
- `{"remaining_sources": [...]}` — additional matches for "show more"
- `{"error": "..."}` — error message

### Nightly Pipeline (GitHub Actions)

`.github/workflows/nightly_data_pipeline.yml` runs daily at midnight UTC:
1. Scrapy spider crawls new Pitchfork reviews
2. Uploads JSONL to S3
3. Runs `create_knowledge_base.py` with `LLM_PROVIDER=GEMINI`, `DB_PROVIDER=CLOUD`
4. GCP credentials are injected via base64-encoded `GCP_CREDENTIALS` secret

## Environment Variables

Required in `.env` at project root:

```
GOOGLE_APPLICATION_CREDENTIALS=gcloud-credentials.json
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
CHROMA_CLOUD_API_KEY=...
CHROMA_CLOUD_TENANT=...
CHROMA_CLOUD_DATABASE=...
DB_PROVIDER=CLOUD          # or LOCAL
APP_LLM_PROVIDER=GEMINI    # or OLLAMA
LLM_PROVIDER=OLLAMA        # for local KB ingestion
```

The `gcloud-credentials.json` file must be at the project root (it's referenced relative to `PROJECT_ROOT` in `config.py`).
