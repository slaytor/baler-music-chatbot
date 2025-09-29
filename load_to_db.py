import pandas as pd
import chromadb
from tqdm import tqdm
import logging


def load_data_to_chroma(
    parquet_path: str,
    collection_name: str,
    db_path: str = "./chroma_db"
):
    """
    Loads embedded review data from a Parquet file into a ChromaDB collection.

    Args:
        parquet_path: Path to the embedded reviews Parquet file.
        collection_name: The name for the ChromaDB collection.
        db_path: The directory to store the ChromaDB database files.
    """
    # --- ADDED: Quiets the noisy logging from ChromaDB ---
    # This will hide the verbose embedding outputs and only show warnings or errors.
    logging.getLogger('chromadb').setLevel(logging.WARNING)

    print(f"Loading embedded data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Convert embedding column from object to a list of lists for Chroma
    df['embedding'] = df['embedding'].apply(list)

    print("Initializing ChromaDB client...")
    # Initialize a persistent client that saves to disk
    client = chromadb.PersistentClient(path=db_path)

    print(f"Creating or getting collection: '{collection_name}'")
    collection = client.get_or_create_collection(name=collection_name)

    # Prepare data for ChromaDB batch insertion
    # ChromaDB requires unique IDs for each entry
    ids = [f"chunk_{i}" for i in range(len(df))]
    documents = df['text_chunk'].tolist()
    embeddings = df['embedding'].tolist()
    metadatas = df.drop(columns=['text_chunk', 'embedding']).to_dict('records')

    # Add data to the collection in batches for memory efficiency
    batch_size = 500
    print(f"Adding {len(df)} documents to ChromaDB in batches of {batch_size}...")

    for i in tqdm(range(0, len(df), batch_size)):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )

    count = collection.count()
    print(f"Database loading complete. Collection '{collection_name}' now contains {count} items.")


if __name__ == "__main__":
    EMBEDDED_DATA_FILE = "reviews_embedded.parquet"
    DB_COLLECTION_NAME = "pitchfork_reviews"

    load_data_to_chroma(EMBEDDED_DATA_FILE, DB_COLLECTION_NAME)
