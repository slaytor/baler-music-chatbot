# This file holds simple, reusable helper functions.

def chunk_text(text: str, chunk_size: int = 4, overlap: int = 1) -> list[str]:
    """Splits text into overlapping chunks of sentences."""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if not sentences:
        return []

    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = ". ".join(sentences[i:i + chunk_size])
        if chunk:
            chunks.append(chunk + '.')
    return chunks
