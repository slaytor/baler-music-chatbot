import json
import time
import chromadb
import pandas as pd
import hashlib
import logging
import torch
from sentence_transformers import SentenceTransformer

from . import config

# --- HELPER FUNCTION FOR DEVICE SELECTION ---
def get_optimal_device():
    """Automatically select the best device for SentenceTransformer."""
    if torch.backends.mps.is_available():
        logging.info("Apple MPS (GPU) is available. Using 'mps'.")
        return 'mps'
    else:
        logging.info("No specialized hardware found. Using 'cpu'.")
        return 'cpu'

class VectorDB:
    """Handles all interactions with the ChromaDB vector database."""

    def __init__(self):
        if config.DB_PROVIDER.upper() == "CLOUD":
            logging.info("Initializing ChromaDB client... Connecting to Chroma Cloud.")
            if not all([config.CHROMA_CLOUD_API_KEY, config.CHROMA_CLOUD_TENANT, config.CHROMA_CLOUD_DATABASE]):
                raise ValueError("CHROMA_CLOUD_API_KEY, CHROMA_CLOUD_TENANT, and CHROMA_CLOUD_DATABASE must be set for CLOUD provider.")
            self.client = chromadb.CloudClient(
                api_key=config.CHROMA_CLOUD_API_KEY,
                tenant=config.CHROMA_CLOUD_TENANT,
                database=config.CHROMA_CLOUD_DATABASE
            )
        else: # Default to LOCAL
            logging.info(f"Initializing ChromaDB client... Connecting to local Docker at {config.CHROMA_HOST}:{config.CHROMA_PORT}")
            self.client = chromadb.HttpClient(
                host=config.CHROMA_HOST, port=config.CHROMA_PORT
            )
        
        self._wait_for_chroma()
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME
        )
        
        device = get_optimal_device()
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
        logging.info("ChromaDB connection successful.")

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

    def get_count(self) -> int:
        """Returns the total number of items in the collection."""
        return self.collection.count()

    def get_processed_urls(self) -> set:
        """Returns a set of all review_urls already in the database."""
        try:
            count = self.get_count()
            logging.info(f"Collection '{self.collection.name}' reports {count} records.")
            if count == 0:
                return set()

            all_urls = set()
            batch_size = 250
            for offset in range(0, count, batch_size):
                logging.info(f"Fetching records from offset {offset}...")
                items = self.collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["metadatas"]
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
        
        # Fetch enough results to cover the offset
        fetch_count = offset + top_k
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=fetch_count
        )
        
        if not results or not results.get("metadatas") or not results["metadatas"][0]:
            return []
            
        all_metadatas = results["metadatas"][0]
        
        # If we don't have enough results to reach the offset, return empty
        if len(all_metadatas) <= offset:
            return []
            
        # Slice the results to return only the requested page
        return all_metadatas[offset : offset + top_k]

    def add_batch(self, enriched_df: pd.DataFrame):
        """
        Embeds and adds a batch of new documents to the database using a robust method.
        """
        if enriched_df.empty:
            return 0

        enriched_df["search_document"] = enriched_df.apply(
            lambda row: f"Tags: {', '.join(row.get('tags', []))}. Review excerpt: {row.get('text_chunk', '')}",
            axis=1,
        )

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
                    documents=enriched_df["search_document"].tolist()[i : i + batch_size]
                )
            except Exception as e:
                logging.error(f"Error upserting batch {i} to ChromaDB: {e}")

        return len(enriched_df)
