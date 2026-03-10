import hashlib
import json
import logging
import re
import time

import chromadb
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from . import config


# --- HELPER FUNCTION FOR DEVICE SELECTION ---
def get_optimal_device():
    """Automatically select the best device for SentenceTransformer."""
    if torch.backends.mps.is_available():
        logging.info("Apple MPS (GPU) is available. Using 'mps'.")
        return "mps"
    else:
        logging.info("No specialized hardware found. Using 'cpu'.")
        return "cpu"


class VectorDB:
    """Handles all interactions with the ChromaDB vector database."""

    def __init__(self):
        if config.DB_PROVIDER.upper() == "CLOUD":
            logging.info("Initializing ChromaDB client... Connecting to Chroma Cloud.")
            if not all(
                [
                    config.CHROMA_CLOUD_API_KEY,
                    config.CHROMA_CLOUD_TENANT,
                    config.CHROMA_CLOUD_DATABASE,
                ]
            ):
                raise ValueError(
                    "CHROMA_CLOUD_API_KEY, CHROMA_CLOUD_TENANT, and CHROMA_CLOUD_DATABASE must be set for CLOUD provider."
                )
            self.client = chromadb.CloudClient(
                api_key=config.CHROMA_CLOUD_API_KEY,
                tenant=config.CHROMA_CLOUD_TENANT,
                database=config.CHROMA_CLOUD_DATABASE,
            )
        else:  # Default to LOCAL
            logging.info(
                f"Initializing ChromaDB client... Connecting to local Docker at {config.CHROMA_HOST}:{config.CHROMA_PORT}"
            )
            self.client = chromadb.HttpClient(
                host=config.CHROMA_HOST, port=config.CHROMA_PORT
            )

        self._wait_for_chroma()
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME
        )

        device = get_optimal_device()
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L6-v2", device=device
        )
        logging.info("ChromaDB connection successful.")

        self._build_bm25_index()

    def _wait_for_chroma(self, timeout: int = 60):
        """Waits for the ChromaDB server to be available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                self.client.heartbeat()
                logging.info("ChromaDB server is up.")
                return
            except Exception:
                logging.info("Waiting for ChromaDB server...")
                time.sleep(5)
        raise TimeoutError("ChromaDB server did not start in time.")

    def _build_bm25_index(self):
        """Fetch all documents from ChromaDB and build an in-memory BM25 index."""
        count = self.get_count()
        if count == 0:
            logging.warning("Collection is empty — BM25 index not built.")
            self.bm25_index = None
            self.bm25_corpus_metadatas = []
            return

        logging.info(f"Building BM25 index from {count} documents...")
        all_documents = []
        all_metadatas = []
        batch_size = 250

        for offset in range(0, count, batch_size):
            logging.info(f"Fetching BM25 corpus batch at offset {offset}...")
            items = self.collection.get(
                limit=batch_size,
                offset=offset,
                include=["metadatas", "documents"],
            )
            if items and items.get("documents"):
                all_documents.extend(items["documents"])
                all_metadatas.extend(items["metadatas"])

        tokenized_corpus = [re.findall(r"[a-z0-9]+(?:'[a-z]+)?(?:-[a-z]+)*", doc.lower()) for doc in all_documents]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.bm25_corpus_metadatas = all_metadatas
        logging.info(f"BM25 index built with {len(all_documents)} documents.")

    def get_count(self) -> int:
        """Returns the total number of items in the collection."""
        return self.collection.count()

    def get_processed_urls(self) -> set:
        """Returns a set of all review_urls already in the database."""
        try:
            count = self.get_count()
            logging.info(
                f"Collection '{self.collection.name}' reports {count} records."
            )
            if count == 0:
                return set()

            all_urls = set()
            batch_size = 250
            for offset in range(0, count, batch_size):
                logging.info(f"Fetching records from offset {offset}...")
                items = self.collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["metadatas"],
                )
                if items and items["metadatas"]:
                    for meta in items["metadatas"]:
                        if "review_url" in meta:
                            all_urls.add(meta["review_url"])

            logging.info(f"Total unique URLs found in database: {len(all_urls)}")
            return all_urls
        except Exception as e:
            logging.error(f"Error getting processed URLs: {e}", exc_info=True)
            return set()

    def search(self, query_text: str, top_k: int, offset: int = 0) -> list:
        """
        Embeds a query and searches the database, returning the top_k results
        with an optional offset.

        Since ChromaDB's query() method does not support an 'offset' parameter,
        we fetch (offset + top_k) results and slice the list manually.
        """
        query_embedding = self.model.encode([query_text]).tolist()

        fetch_count = offset + top_k

        results = self.collection.query(
            query_embeddings=query_embedding, n_results=fetch_count
        )

        if not results or not results.get("metadatas") or not results["metadatas"][0]:
            return []

        all_metadatas = results["metadatas"][0]

        if len(all_metadatas) <= offset:
            return []

        return all_metadatas[offset : offset + top_k]

    def bm25_search(self, query_text: str, top_k: int) -> list:
        """Returns top_k metadata records ranked by BM25 keyword score."""
        if self.bm25_index is None:
            return []
        tokenized_query = re.findall(r"[a-z0-9]+(?:'[a-z]+)?(?:-[a-z]+)*", query_text.lower())
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]
        return [self.bm25_corpus_metadatas[i] for i in top_indices]

    def hybrid_search(self, query_text: str, top_k: int = 50) -> list:
        """
        Two-layer retrieval: BM25 (sparse) + ChromaDB vector (dense), fused with
        Reciprocal Rank Fusion (RRF). Returns top_k fused candidates.
        """
        CANDIDATE_COUNT = 100
        RRF_K = 60

        # --- BM25 layer ---
        bm25_results = self.bm25_search(query_text, CANDIDATE_COUNT)

        # --- Dense vector layer ---
        query_embedding = self.model.encode([query_text]).tolist()
        chroma_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=CANDIDATE_COUNT,
            include=["metadatas"],
        )
        chroma_metadatas = (
            chroma_results["metadatas"][0]
            if chroma_results and chroma_results.get("metadatas")
            else []
        )

        # --- RRF fusion ---
        def chunk_key(meta: dict) -> str:
            return hashlib.sha256(
                f"{meta.get('review_url', '')}{meta.get('text_chunk', '')}".encode()
            ).hexdigest()

        rrf_scores: dict[str, float] = {}
        meta_lookup: dict[str, dict] = {}

        for rank, meta in enumerate(bm25_results):
            key = chunk_key(meta)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (RRF_K + rank + 1)
            meta_lookup[key] = meta

        for rank, meta in enumerate(chroma_metadatas):
            key = chunk_key(meta)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (RRF_K + rank + 1)
            meta_lookup[key] = meta

        sorted_keys = sorted(
            rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True
        )

        # Keep only the highest-scoring chunk per album so long reviews don't
        # crowd out albums with fewer chunks in the candidate pool.
        seen_urls: set[str] = set()
        deduped: list[dict] = []
        for k in sorted_keys:
            meta = meta_lookup[k]
            url = meta.get("review_url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped.append(meta)
            if len(deduped) >= top_k:
                break

        return deduped

    def rerank(self, query_text: str, candidates: list) -> list:
        """
        Cross-encoder re-ranking over (query, text_chunk) pairs, with a genre
        coherence boost. Deduplicates by review_url, keeping the highest-scoring
        chunk per album.
        """
        if not candidates:
            return []

        pairs = [(query_text, meta.get("text_chunk", "")) for meta in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Initial sort to identify top genres
        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        top_genres: set[str] = set()
        seen: set[str] = set()
        for _score, meta in scored:
            url = meta.get("review_url")
            if url and url not in seen:
                seen.add(url)
                try:
                    genres = json.loads(meta.get("artist_genres", "[]"))
                    top_genres.update(g.lower() for g in genres)
                except (json.JSONDecodeError, TypeError):
                    pass
            if len(seen) >= 5:
                break

        # Re-score with a gentle genre coherence boost (0.1 per overlapping genre)
        def boosted(score: float, meta: dict) -> float:
            try:
                genres = {g.lower() for g in json.loads(meta.get("artist_genres", "[]"))}
            except (json.JSONDecodeError, TypeError):
                genres = set()
            return score + len(genres & top_genres) * 0.1

        scored_boosted = sorted(
            ((boosted(s, m), m) for s, m in zip(scores, candidates)),
            key=lambda x: x[0],
            reverse=True,
        )

        seen_urls: set[str] = set()
        result = []
        for _score, meta in scored_boosted:
            url = meta.get("review_url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                result.append(meta)

        return result

    def apply_exclusion_filters(self, candidates: list, filters: dict) -> list:
        """
        Post-retrieval filter that removes candidates matching user-specified exclusions.
        Checks artist_genres, artist, and release_year against the parsed filter dict.
        """
        exclude_genres = [g.lower() for g in filters.get("exclude_genres", [])]
        exclude_artists = [a.lower() for a in filters.get("exclude_artists", [])]
        max_year = filters.get("max_year")
        min_year = filters.get("min_year")

        if not any([exclude_genres, exclude_artists, max_year, min_year]):
            return candidates

        filtered = []
        for meta in candidates:
            # Genre filter — substring match against artist_genres and tags
            if exclude_genres:
                try:
                    genres = [g.lower() for g in json.loads(meta.get("artist_genres", "[]"))]
                except (json.JSONDecodeError, TypeError):
                    genres = []
                try:
                    tags = [t.lower() for t in json.loads(meta.get("tags", "[]"))]
                except (json.JSONDecodeError, TypeError):
                    tags = []
                all_terms = genres + tags
                if any(eg in term for eg in exclude_genres for term in all_terms):
                    continue

            # Artist filter — exclude albums by the named artist
            if exclude_artists:
                artist = meta.get("artist", "").lower()
                if any(ea in artist or artist in ea for ea in exclude_artists):
                    continue

            # Year filter
            try:
                year = int(meta.get("release_year", 0))
            except (ValueError, TypeError):
                year = 0
            if max_year and year and year > max_year:
                continue
            if min_year and year and year < min_year:
                continue

            filtered.append(meta)

        return filtered

    def expand_with_related_artists(self, query_text: str, top_results: list) -> list:
        """
        Takes the top results, extracts their related_artists, and fetches additional
        candidate albums by those artists via vector search. Returns candidates not
        already in top_results, for use as discovery/"show more" entries.
        """
        top_urls = {m.get("review_url") for m in top_results}
        top_artists = {m.get("artist", "").lower() for m in top_results}

        related_names: set[str] = set()
        for meta in top_results[:5]:
            try:
                related = json.loads(meta.get("related_artists", "[]"))
                for name in related[:5]:
                    if name.lower() not in top_artists:
                        related_names.add(name)
            except (json.JSONDecodeError, TypeError):
                pass

        if not related_names:
            return []

        query_embedding = self.model.encode([query_text]).tolist()
        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=50,
                where={"artist": {"$in": list(related_names)}},
                include=["metadatas"],
            )
            candidates = (
                results["metadatas"][0]
                if results and results.get("metadatas")
                else []
            )
        except Exception as e:
            logging.warning(f"Related artist expansion failed: {e}")
            return []

        return [m for m in candidates if m.get("review_url") not in top_urls]

    def add_batch(self, enriched_df: pd.DataFrame):
        """
        Embeds and adds a batch of new documents to the database using a robust method.
        """
        if enriched_df.empty:
            return 0

        def _build_search_document(row) -> str:
            parts = []
            genres = row.get("artist_genres", "[]")
            if isinstance(genres, str):
                try:
                    genres = json.loads(genres)
                except (json.JSONDecodeError, TypeError):
                    genres = []
            if genres:
                parts.append(f"Genres: {', '.join(genres)}.")
            tags = row.get("tags", [])
            if tags:
                parts.append(f"Tags: {', '.join(tags)}.")
            parts.append(f"Review excerpt: {row.get('text_chunk', '')}")
            return " ".join(parts)

        enriched_df["search_document"] = enriched_df.apply(_build_search_document, axis=1)

        logging.info(f"Embedding {len(enriched_df)} new search documents...")
        embeddings = self.model.encode(
            enriched_df["search_document"].tolist(), show_progress_bar=True
        )

        ids = [
            hashlib.sha256(
                f"{row.get('review_url', '')}{row.get('text_chunk', '')}".encode()
            ).hexdigest()
            for _, row in enriched_df.iterrows()
        ]

        metadatas = []
        for _, row in enriched_df.iterrows():
            meta = row.to_dict()
            meta.pop("search_document", None)
            meta["tags"] = json.dumps(meta.get("tags", []))

            for key, value in meta.items():
                if value is None:
                    meta[key] = "N/A"

            metadatas.append(meta)

        batch_size = 100
        for i in range(0, len(enriched_df), batch_size):
            try:
                self.collection.upsert(
                    ids=ids[i : i + batch_size],
                    embeddings=embeddings[i : i + batch_size].tolist(),
                    metadatas=metadatas[i : i + batch_size],
                    documents=enriched_df["search_document"].tolist()[
                        i : i + batch_size
                    ],
                )
            except Exception as e:
                logging.error(f"Error upserting batch {i} to ChromaDB: {e}")

        return len(enriched_df)
