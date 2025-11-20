import json
import pandas as pd
import chromadb
from pathlib import Path

# --- CONFIGURATION ---
# This script connects to your ORIGINAL local database folder.
LOCAL_DB_PATH = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "pitchfork_reviews"
OUTPUT_FILE = Path(__file__).parent / "enriched_data.jsonl"

def export_local_data():
    """
    Connects to the local file-based ChromaDB, extracts all records,
    and saves them to a JSONL file.
    """
    print(f"Connecting to local ChromaDB at: {LOCAL_DB_PATH}")
    try:
        client = chromadb.PersistentClient(path=LOCAL_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"FATAL: Could not connect to the local database. Have you run the old version before? Error: {e}")
        return

    count = collection.count()
    if count == 0:
        print("The local database is empty. Nothing to export.")
        return

    print(f"Found {count} items in the local database. Fetching all records...")

    # Fetch all records from the collection
    # We fetch in batches to be safe, though for 120MB 'get()' is likely fine.
    batch_size = 1000
    all_metadatas = []
    for offset in range(0, count, batch_size):
        results = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"]
        )
        if results['metadatas']:
            all_metadatas.extend(results['metadatas'])

    if not all_metadatas:
        print("Could not retrieve any metadata from the database.")
        return

    print(f"Successfully fetched {len(all_metadatas)} records.")

    # The metadata already contains everything we need.
    # We just need to ensure the 'tags' are correctly formatted as a list.
    for item in all_metadatas:
        if 'tags' in item and isinstance(item['tags'], str):
            try:
                item['tags'] = json.loads(item['tags'])
            except json.JSONDecodeError:
                print(f"Warning: Could not parse tags for item: {item.get('review_url')}")
                item['tags'] = []

    # Convert to a pandas DataFrame and save to JSONL
    df = pd.DataFrame(all_metadatas)
    
    print(f"Saving {len(df)} enriched records to {OUTPUT_FILE}...")
    df.to_json(OUTPUT_FILE, orient='records', lines=True)

    print("\nExport complete.")
    print(f"Your enriched data has been saved to: {OUTPUT_FILE}")
    print("You can now proceed to the import step.")


if __name__ == "__main__":
    export_local_data()
