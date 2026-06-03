"""Milestone 4 — Embed chunks and upsert into ChromaDB.

Builds chunks via chunk.build_chunks(), encodes them with all-MiniLM-L6-v2,
and upserts into a persistent ChromaDB collection with metadata.

Run:  python embed.py
"""

from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

from chunk import build_chunks

EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "chroma_db"
COLLECTION = "qml_papers"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazily load (and cache) the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Cosine space matches how we reason about distances (<0.4 = strong match).
    return client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def main() -> None:
    chunks = build_chunks()
    print(f"\nEmbedding {len(chunks)} chunks with {EMBED_MODEL}...")

    model = get_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True
    ).tolist()

    collection = get_collection()
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {k: c[k] for k in ("source", "section", "cluster", "year")}
            for c in chunks
        ],
    )

    print(f"Upserted {collection.count()} chunks into "
          f"'{COLLECTION}' at ./{CHROMA_PATH}")


if __name__ == "__main__":
    main()
