"""Tests for hybrid retrieval with RRF."""

from src.retriever import reciprocal_rank_fusion


def test_rrf_combines_two_rankings():
    """RRF should merge two ranked lists, boosting items that appear in both."""
    ranking_a = ["doc1", "doc2", "doc3"]  # doc1 is best in A
    ranking_b = ["doc3", "doc1", "doc2"]  # doc3 is best in B

    fused = reciprocal_rank_fusion([ranking_a, ranking_b], k=60)
    # doc1 appears at rank 1 and rank 2 → strong combined score
    # doc3 appears at rank 3 and rank 1 → strong combined score
    assert len(fused) == 3
    # All docs should be present
    assert set(fused) == {"doc1", "doc2", "doc3"}


def test_rrf_deduplicates():
    """Same doc in both lists should appear only once in output."""
    ranking_a = ["doc1", "doc2"]
    ranking_b = ["doc1", "doc3"]

    fused = reciprocal_rank_fusion([ranking_a, ranking_b], k=60)
    assert fused.count("doc1") == 1


def test_rrf_empty_input():
    fused = reciprocal_rank_fusion([], k=60)
    assert fused == []


def test_rrf_single_ranking():
    ranking = ["doc1", "doc2", "doc3"]
    fused = reciprocal_rank_fusion([ranking], k=60)
    assert fused == ranking
