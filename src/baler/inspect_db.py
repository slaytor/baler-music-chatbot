import chromadb
import logging
from pathlib import Path
import json
import pprint

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "pitchfork_reviews"


def inspect_database_sample(num_samples: int = 5):
    """Connects to ChromaDB and prints a sample of items from the collection."""
    logging.getLogger('chromadb').setLevel(logging.WARNING)

    print(f"Connecting to database at '{DB_PATH}'...")
    try:
        client = chromadb.PersistentClient(path=str(DB_PATH))
        print(f"Getting collection '{COLLECTION_NAME}'...")
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        print(f"Collection contains {count} items.")

        if count == 0:
            print("Database is empty.")
            return

        num_to_get = min(num_samples, count)
        print(f"\nRetrieving {num_to_get} sample items...")

        # Retrieve items including documents and metadata
        # peek() is an efficient way to get a small sample
        results = collection.peek(limit=num_to_get)

        print("\n--- Sample Items ---")
        for i in range(len(results['ids'])):
            print(f"\nItem ID: {results['ids'][i]}")

            # The document is the enriched text with tags + excerpt
            print("Document (Indexed Text):")
            print(f"  {results['documents'][i]}")

            # Metadata contains the original chunk, tags, and review info
            print("Metadata:")
            metadata = results['metadatas'][i]
            # Try to parse the tags back into a list for pretty printing
            try:
                metadata['tags'] = json.loads(metadata.get('tags', '[]'))
            except json.JSONDecodeError:
                pass # Keep tags as string if parsing fails
            pprint.pprint(metadata, indent=2) # Pretty print the metadata dictionary
            print("-" * 20)

    except Exception as e:
        print(f"\nError accessing the database: {e}")


if __name__ == "__main__":
    inspect_database_sample(num_samples=15) # Get 5 samples by default
