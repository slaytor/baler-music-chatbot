# Get 'Forked 

### [Visit the Live App: get-forked.org](https://get-forked.org)

**Get 'Forked** (internally codenamed *Baler*) is an AI-powered music sommelier that provides nuanced, context-aware album recommendations based on Pitchfork reviews.

Unlike standard recommendation algorithms that rely on collaborative filtering ("people who liked X also liked Y"), Get 'Forked uses **Retrieval-Augmented Generation (RAG)** to understand the *vibe*, *texture*, and *emotional quality* of music as described by professional critics.

---

## Key Features

*   **Natural Language Search:** Ask for "dreamy shoegaze for a rainy tuesday" or "aggressive punk to clean my apartment to."
*   **Deep Knowledge Base:** Powered by a vector database containing tens of thousands of review fragments.
*   **Spotify Integration:** One-click access to listen to recommended albums directly on Spotify.
*   **Infinite Discovery:** "Show more" functionality allows for deep diving into specific sub-genres.
*   **Pretentious Mode:** Dynamic, randomized suggestion prompts that mimic the hyper-specific language of music nerds.

---

## Tech Stack

### Core Application
*   **Framework:** Python 3.12, FastAPI, Uvicorn
*   **Frontend:** Vanilla JS, Tailwind CSS (No build step required)
*   **Deployment:** Docker, Google Cloud Run (Serverless)

### AI & Data
*   **LLM:** Google Gemini 1.5 Flash (via `google-auth`)
*   **Vector Database:** ChromaDB (Cloud)
*   **Embeddings:** `all-MiniLM-L6-v2` (SentenceTransformers)
*   **Music Data:** Spotify Web API

### Automation & Pipeline
*   **Scraping:** Scrapy & Playwright
*   **Orchestration:** GitHub Actions
*   **Storage:** AWS S3 (Raw data lake)

---

## The Nightly Data Pipeline

One of the core strengths of this project is its self-updating knowledge base. The application stays current without manual intervention through a fully automated pipeline:

1.  **Trigger:** Every night at 08:00 UTC, a **GitHub Action** spins up.
2.  **Scrape:** It runs a custom **Scrapy/Playwright** spider to fetch the latest reviews published on Pitchfork that day.
3.  **Archive:** Raw JSONL data is uploaded to an **AWS S3** bucket for long-term storage and versioning.
4.  **Process:** The pipeline downloads the new data, cleans it, and chunks the text.
5.  **Embed:** It generates vector embeddings for the new reviews using `SentenceTransformers`.
6.  **Update:** The new vectors are upserted into the **Chroma Cloud** database, making them immediately available to the live application.

---

## Local Development

To run the application locally:

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
