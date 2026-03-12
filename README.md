# Get 'Forked

### [Visit the Live App: get-forked.org](https://get-forked.org)

**Get 'Forked** (internally codenamed *Baler*) is an AI-powered music sommelier that provides nuanced, context-aware album recommendations based on Pitchfork reviews.

Unlike standard recommendation algorithms that rely on collaborative filtering ("people who liked X also liked Y"), Get 'Forked uses **Retrieval-Augmented Generation (RAG)** to understand the *vibe*, *texture*, and *emotional quality* of music as described by professional critics.

---

## Key Features

*   **Natural Language Search:** Ask for "dreamy shoegaze for a rainy tuesday" or "aggressive punk to clean my apartment to."
*   **Deep Knowledge Base:** ~27,000 Pitchfork album reviews, chunked and indexed as ~213,000 vector embeddings.
*   **Hybrid Retrieval:** BM25 sparse search + dense vector search fused with Reciprocal Rank Fusion, then reranked by a cross-encoder for precision.
*   **Related Artist Expansion:** Top results seed a second-pass retrieval over similar artists for deeper discovery.
*   **Spotify Integration:** One-click access to listen to recommended albums directly on Spotify.
*   **Show More:** Additional matches surfaced beyond the main recommendation for deeper genre exploration.

---

## Tech Stack

### Core Application
*   **Framework:** Python 3.12, FastAPI, Uvicorn
*   **Frontend:** Vanilla JS, Tailwind CSS (no build step)
*   **Deployment:** Docker, Google Cloud Run (serverless)

### AI & Retrieval
*   **LLM:** Google Gemini 2.0 Flash (via `google-auth`)
*   **Vector Database:** ChromaDB Cloud
*   **Embeddings:** `all-MiniLM-L6-v2` (SentenceTransformers)
*   **Sparse Search:** BM25 (in-memory, lazy-loaded on startup)
*   **Reranking:** `cross-encoder/ms-marco-MiniLM-L6-v2`
*   **Music Metadata:** Last.fm API (genres, related artists), Spotify Web API (album links)

---

## Knowledge Base

The knowledge base is a static snapshot of ~27,000 Pitchfork album reviews scraped using a Scrapy/Playwright spider. Reviews are chunked, tagged with LLM-generated semantic descriptors, enriched with Last.fm genre and related artist data, and embedded into ChromaDB Cloud.

The scraping and ingestion pipeline (Scrapy, GitHub Actions, AWS S3) exists in the repo but is not actively running — the knowledge base reflects the state as of early 2026.

---

## Local Development

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/yourusername/baler-music-chatbot.git
    cd baler-music-chatbot
    ```

2.  **Set up environment variables:**
    Create a `.env` file with your credentials (see `config.py` for required fields).

3.  **Run with Docker Compose:**
    ```bash
    docker-compose up --build
    ```

4.  **Access the app:**
    Open `http://localhost:8080` in your browser.

---

*Built by Sam Taylor.*
