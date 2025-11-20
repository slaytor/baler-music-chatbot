import json
import logging
import pprint

from .database import VectorDB


def check_database_count(num_samples: int = 0):
    """Connects to ChromaDB via the service class and prints the item count."""
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    try:
        db = VectorDB()
        count = db.get_count()
        print("\n-----------------------------------------")
        print(f"Success! Your database contains {count} items.")
        print("-----------------------------------------")

        if num_samples > 0 and count > 0:
            num_to_get = min(num_samples, count)
            print(f"\nRetrieving {num_to_get} sample items...")

            # Use the raw client 'peek' for this utility
            results = db.collection.peek(limit=num_to_get)

            print("\n--- Sample Items ---")
            for i in range(len(results["ids"])):
                print(f"\nItem ID: {results['ids'][i]}")
                print("Document (Indexed Text):")
                print(f"  {results['documents'][i]}")
                print("Metadata:")
                metadata = results["metadatas"][i]
                try:
                    metadata["tags"] = json.loads(metadata.get("tags", "[]"))
                except json.JSONDecodeError:
                    pass
                pprint.pprint(metadata, indent=2)
                print("-" * 20)

    except Exception as e:
        print(f"\nError accessing the database: {e}")


if __name__ == "__main__":
    # You can run this to just get the count:
    # poetry run python -m baler.check_db
    # Or add an argument to see samples:
    # poetry run python -m baler.check_db 5
    import sys

    samples = 0
    if len(sys.argv) > 1:
        try:
            samples = int(sys.argv[1])
        except ValueError:
            print("Usage: python -m baler.check_db [num_samples]")

    check_database_count(num_samples=samples)
