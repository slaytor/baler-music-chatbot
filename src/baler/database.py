import json
import time
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from . import config


class VectorDB:
    """Handles all interactions with the ChromaDB vector database."""

    def __init__(self):
        print(
            f"Initializing ChromaDB client... Connecting to {config.CHROMA_HOST}:{config.CHROMA_PORT}"
        )
        self.client = chromadb.HttpClient(
            host=config.CHROMA_HOST, port=config.CHROMA_PORT
        )
        self._wait_for_chroma()
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME
        )
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        print("ChromaDB connection successful.")

    def _wait_for_chroma(self, timeout: int = 60):
        """Waits for the ChromaDB server to be available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                self.client.heartbeat()
                print("ChromaDB server is up.")
                return
            except Exception:
                print("Waiting for ChromaDB server...")
                time.sleep(5)
        raise TimeoutError("ChromaDB server did not start in time.")

    def get_count(self) -> int:
        """Returns the total number of items in the collection."""
        return self.collection.count()

    def get_processed_urls(self) -> set:
        """Returns a set of all review_urls already in the database."""
        try:
            batch_size = 1000
            count = self.get_count()
            all_urls = set()
            for offset in range(0, count, batch_size):
                items = self.collection.get(
                    limit=batch_size, offset=offset, include=["metadatas"]
                )
                if items["metadatas"]:
                    for meta in items["metadatas"]:
                        if "review_url" in meta:
                            all_urls.add(meta["review_url"])
            return all_urls
        except Exception as e:
            print(f"Error getting processed URLs (DB might be empty): {e}")
        return set()

    def search(self, query_text: str, top_k: int) -> list:
        """
        Embeds a query and searches the database, returning the top_k results.
        """
        query_embedding = self.model.encode([query_text]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding, n_results=top_k
        )
        if not results or not results.get("metadatas") or not results["metadatas"][0]:
            return []
        return [metadata for metadata in results["metadatas"][0]]

    def add_batch(self, enriched_df: pd.DataFrame):
        """
        Embeds and adds a batch of new documents to the database.
        """
        if enriched_df.empty:
            return 0

        enriched_df["search_document"] = enriched_df.apply(
            lambda row: f"Tags: {', '.join(row.get('tags', []))}. Review excerpt: {row.get('text_chunk', '')}",
            axis=1,
        )

        print(f"Embedding {len(enriched_df)} new search documents...")
        embeddings = self.model.encode(
            enriched_df["search_document"].tolist(), show_progress_bar=True
        )

        current_count = self.get_count()
        ids = [f"chunk_{current_count + i}" for i in range(len(enriched_df))]

        metadatas = []
        for _, row in enriched_df.iterrows():
            meta = row.to_dict()
            meta.pop("search_document", None)
            meta["tags"] = json.dumps(meta.get("tags", []))
            
            # --- SANITIZATION STEP ---
            # Ensure all metadata values are valid types for ChromaDB.
            for key, value in meta.items():
                if value is None:
                    meta[key] = "N/A" # Replace None with a default string
            
            metadatas.append(meta)

        batch_size = 500
        for i in range(0, len(enriched_df), batch_size):
            try:
                self.collection.add(
                    ids=ids[i : i + batch_size],
                    embeddings=embeddings[i : i + batch_size].tolist(),
                    metadatas=metadatas[i : i + batch_size],
                )
            except Exception as e:
                print(f"Error adding batch {i} to ChromaDB: {e}")

        return len(enriched_df)
