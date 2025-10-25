import chromadb
import logging

# --- CONFIGURATION ---
DB_PATH = "./chroma_db"
COLLECTION_NAME = "pitchfork_reviews"


def check_database_count():
    """Connects to ChromaDB and prints the number of items in the collection."""
    # Hide verbose logging from ChromaDB
    logging.getLogger('chromadb').setLevel(logging.WARNING)

    print(f"Connecting to database at '{DB_PATH}'...")
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        print("\n-----------------------------------------")
        print(f"Success! Your database contains {count} items.")
        print("-----------------------------------------")
    except Exception as e:
        print(f"\nCould not connect to the database. It might not exist yet. Error: {e}")


if __name__ == "__main__":
    check_database_count()
