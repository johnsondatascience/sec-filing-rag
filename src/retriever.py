"""Hybrid retrieval: dense + BM25 with Reciprocal Rank Fusion and reranking."""

from __future__ import annotations

from collections import defaultdict

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.chunker import Chunk
from src.config import (
    RERANKER_MODEL,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    RRF_K,
)
from src.embedder import embed_query
from src.indexer import tokenize


_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[str]:
    """Combine multiple ranked lists using RRF.

    Score for doc d = sum over rankings of 1 / (k + rank(d))
    Returns doc IDs sorted by fused score (descending).
    """
    if not rankings:
        return []

    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    return sorted(scores, key=scores.__getitem__, reverse=True)


def search_dense(
    query: str,
    collection: chromadb.Collection,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[str]:
    """Dense vector search via ChromaDB. Returns ranked chunk IDs."""
    query_emb = embed_query(query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents"],
    )
    return results["ids"][0]


def search_bm25(
    query: str,
    bm25: BM25Okapi,
    chunks: list[Chunk],
    top_k: int = RETRIEVAL_TOP_K,
) -> list[str]:
    """BM25 sparse search. Returns ranked chunk IDs."""
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i].chunk_id for i in top_indices if scores[i] > 0]


def hybrid_search(
    query: str,
    collection: chromadb.Collection,
    bm25: BM25Okapi,
    chunks: list[Chunk],
    top_k: int = RETRIEVAL_TOP_K,
) -> list[str]:
    """Hybrid search combining dense + BM25 via RRF. Returns chunk IDs."""
    dense_ids = search_dense(query, collection, top_k=top_k)
    bm25_ids = search_bm25(query, bm25, chunks, top_k=top_k)
    fused_ids = reciprocal_rank_fusion([dense_ids, bm25_ids], k=RRF_K)
    return fused_ids[:top_k]


def rerank(
    query: str,
    chunk_ids: list[str],
    chunk_lookup: dict[str, Chunk],
    top_k: int = RERANK_TOP_K,
) -> list[str]:
    """Re-score candidates with cross-encoder and return top_k."""
    if not chunk_ids:
        return []

    reranker = get_reranker()
    pairs = [(query, chunk_lookup[cid].text) for cid in chunk_ids if cid in chunk_lookup]
    valid_ids = [cid for cid in chunk_ids if cid in chunk_lookup]

    if not pairs:
        return []

    scores = reranker.predict(pairs)
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [valid_ids[i] for i in ranked_indices]


def retrieve(
    query: str,
    collection: chromadb.Collection,
    bm25: BM25Okapi,
    chunks: list[Chunk],
) -> list[Chunk]:
    """Full retrieval pipeline: hybrid search → rerank → return chunks."""
    chunk_lookup = {c.chunk_id: c for c in chunks}
    candidate_ids = hybrid_search(query, collection, bm25, chunks)
    reranked_ids = rerank(query, candidate_ids, chunk_lookup)
    return [chunk_lookup[cid] for cid in reranked_ids]
