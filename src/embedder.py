"""Local embedding model using sentence-transformers on GPU."""

from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load embedding model to GPU."""
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Encode texts into dense embeddings.

    Returns list of embedding vectors as plain lists (ChromaDB compatible).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Encode a single query string."""
    model = get_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()
