import chromadb
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from . import config # Import our centralized config


class VectorDB:
    """Handles all interactions with the ChromaDB vector database."""

    def __init__(self):
        print("Initializing ChromaDB connection...")
        self.client = chromadb.PersistentClient(path=str(config.DB_PATH))
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME
        )
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        print("ChromaDB connection successful.")

    def get_count(self) -> int:
        """Returns the total number of items in the collection."""
        return self.collection.count()

    def get_processed_urls(self) -> set:
        """Returns a set of all review_urls already in the database."""
        try:
            items = self.collection.get(include=["metadatas"])
            if items['metadatas']:
                return {item['review_url'] for item in items['metadatas']}
        except Exception as e:
            print(f"Error getting processed URLs (DB might be empty): {e}")
        return set()

    def search(self, query_text: str, top_k: int) -> list:
        """
        Embeds a query and searches the database, returning the top_k results.
        """
        query_embedding = self.model.encode([query_text]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        if not results or not results.get('metadatas') or not results['metadatas'][0]:
            return []

        # Return the metadata, which contains the original chunk and review info
        return [metadata for metadata in results['metadatas'][0]]

    def add_batch(self, enriched_df: pd.DataFrame):
        """
        Embeds and adds a batch of new documents to the database.

        Args:
            enriched_df: A DataFrame with columns ['artist', 'album_title',
                         'score', 'review_url', 'text_chunk', 'tags']
        """
        if enriched_df.empty:
            return 0

        # 1. Create the searchable document
        enriched_df['search_document'] = enriched_df.apply(
            lambda row: f"Tags: {', '.join(row['tags'])}. Review excerpt: {row['text_chunk']}",
            axis=1
        )

        # 2. Embed the documents
        print(f"Embedding {len(enriched_df)} new search documents...")
        embeddings = self.model.encode(
            enriched_df['search_document'].tolist(),
            show_progress_bar=True
        )

        # 3. Prepare data for ChromaDB
        current_count = self.get_count()
        ids = [f"chunk_{current_count + i}" for i in range(len(enriched_df))]
        documents = enriched_df['search_document'].tolist()

        metadatas = []
        for _, row in enriched_df.iterrows():
            meta = row.to_dict()
            meta.pop('search_document', None)
            meta['tags'] = json.dumps(meta.get('tags', []))
            for key in ['artist', 'album_title', 'score', 'review_url', 'text_chunk']:
                if key not in meta: meta[key] = "N/A"
            metadatas.append(meta)

        # 4. Add to collection in batches
        batch_size = 100
        for i in range(0, len(enriched_df), batch_size):
            try:
                self.collection.add(
                    ids=ids[i:i + batch_size],
                    embeddings=embeddings[i:i + batch_size].tolist(),
                    metadatas=metadatas[i:i + batch_size]
                )
            except Exception as e:
                print(f"Error adding batch {i} to ChromaDB: {e}")

        return len(enriched_df)
