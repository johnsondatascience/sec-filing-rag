"""Build ChromaDB vector index and BM25 sparse index from chunks."""

from __future__ import annotations

import json
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from src.chunker import Chunk
from src.config import CHROMA_COLLECTION_NAME, CHROMA_DIR, BM25_K1, BM25_B
from src.embedder import embed_texts


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [w for w in text.split() if len(w) > 1]


def build_chroma_index(chunks: list[Chunk]) -> chromadb.Collection:
    """Create or replace ChromaDB collection from chunks."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if present
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch embed and insert
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        embeddings = embed_texts(texts)
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"company": c.company, "year": c.year, "section": c.section, "strategy": c.strategy}
                for c in batch
            ],
        )

    return collection


def build_bm25_index(chunks: list[Chunk]) -> tuple[BM25Okapi, list[Chunk]]:
    """Build BM25 index over chunk texts.

    Returns the BM25 object and the aligned chunk list (needed for lookup).
    """
    tokenized = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized, k1=BM25_K1, b=BM25_B)
    return bm25, chunks


def save_chunks_metadata(chunks: list[Chunk], path: Path) -> None:
    """Save chunk metadata to JSON for reproducibility."""
    data = [
        {
            "chunk_id": c.chunk_id,
            "company": c.company,
            "year": c.year,
            "section": c.section,
            "strategy": c.strategy,
            "text_preview": c.text[:200],
        }
        for c in chunks
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
