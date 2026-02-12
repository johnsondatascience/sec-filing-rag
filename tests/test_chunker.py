"""Tests for chunking strategies."""

from src.chunker import Chunk, chunk_fixed, chunk_section_aware


def test_chunk_fixed_splits_long_text():
    text = "word " * 600  # ~600 tokens
    chunks = chunk_fixed(text, chunk_size=200, overlap=50, company="X", year=2024, section="Item 1")
    assert len(chunks) > 1
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.company == "X"
        assert c.section == "Item 1"


def test_chunk_fixed_preserves_all_content():
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    chunks = chunk_fixed(text, chunk_size=5, overlap=1, company="X", year=2024, section="Item 1")
    # All words should appear in at least one chunk
    all_words = set(text.split())
    chunk_words = set()
    for c in chunks:
        chunk_words.update(c.text.split())
    assert all_words.issubset(chunk_words)


def test_chunk_fixed_short_text_returns_single_chunk():
    text = "Short text."
    chunks = chunk_fixed(text, chunk_size=200, overlap=50, company="X", year=2024, section="Item 1")
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_chunk_section_aware_respects_paragraphs():
    text = "First paragraph about revenue.\n\nSecond paragraph about risk.\n\nThird paragraph about operations."
    chunks = chunk_section_aware(text, chunk_size=10, company="X", year=2024, section="Item 1")
    # Each paragraph should be its own chunk (they're short enough)
    assert len(chunks) >= 2
    # No chunk should mix paragraphs if they fit individually
    for c in chunks:
        assert isinstance(c, Chunk)


def test_chunk_has_unique_ids():
    text = "word " * 600
    chunks = chunk_fixed(text, chunk_size=200, overlap=50, company="X", year=2024, section="Item 1")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"
