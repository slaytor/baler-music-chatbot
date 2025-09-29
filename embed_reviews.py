import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Use a pre-trained sentence-transformer model
# This model is small, fast, and great for general-purpose semantic search
MODEL_NAME = 'all-MiniLM-L6-v2'


def chunk_text(text: str, chunk_size: int = 4, overlap: int = 1) -> list[str]:
    """
    Splits a long text into smaller, overlapping chunks of sentences.

    Args:
        text: The review text to be chunked.
        chunk_size: The number of sentences in each chunk.
        overlap: The number of sentences to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    # A simple sentence splitter based on periods.
    # More advanced libraries like spaCy or NLTK could be used for better accuracy.
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if not sentences:
        return []

    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = ". ".join(sentences[i:i + chunk_size])
        if chunk:
            chunks.append(chunk + '.')
    return chunks


def process_and_embed_reviews(input_path: str, output_path: str):
    """
    Loads cleaned reviews, chunks the text, creates vector embeddings,
    and saves the final AI-ready data.

    Args:
        input_path: Path to the cleaned reviews Parquet file.
        output_path: Path to save the final embedded data Parquet file.
    """
    print("Loading cleaned data...")
    df = pd.read_parquet(input_path)

    print(f"Loading embedding model '{MODEL_NAME}'...")
    # Initialize the sentence-transformer model
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")

    processed_data = []

    # Use tqdm for a nice progress bar
    print("Processing and embedding reviews...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        # 1. Chunk the review text
        text_chunks = chunk_text(row['review_text'])
        if not text_chunks:
            continue

        # 2. Embed all chunks in a single batch for efficiency
        embeddings = model.encode(text_chunks, convert_to_tensor=False)

        # 3. Create a record for each chunk with its embedding and metadata
        for chunk, embedding in zip(text_chunks, embeddings):
            processed_data.append({
                'artist': row['artist'],
                'album_title': row['album_title'],
                'score': row['score'],
                'is_best_new_music': row['is_best_new_music'],
                'review_url': row['review_url'],
                'release_year': row['release_year'],
                'author': row['author'],
                'text_chunk': chunk,
                'embedding': embedding
            })

    print(f"Created {len(processed_data)} text chunks.")

    # Convert the processed data to a new DataFrame
    embedded_df = pd.DataFrame(processed_data)

    print(f"Saving embedded data to {output_path}...")
    embedded_df.to_parquet(output_path)
    print("Embedding complete!")


if __name__ == "__main__":
    CLEANED_DATA_FILE = "reviews_cleaned.parquet"
    EMBEDDED_DATA_FILE = "reviews_embedded.parquet"

    process_and_embed_reviews(CLEANED_DATA_FILE, EMBEDDED_DATA_FILE)
