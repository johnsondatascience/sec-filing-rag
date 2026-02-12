"""Tests for citation generation and post-processing."""

from src.generator import build_prompt, extract_citations, format_source_label
from src.chunker import Chunk


def test_build_prompt_includes_sources_and_query():
    chunks = [
        Chunk(text="Revenue was $2B.", company="ACME", year=2024, section="Item 7"),
        Chunk(text="Risk: competition.", company="ACME", year=2024, section="Item 1A"),
    ]
    prompt = build_prompt("What was revenue?", chunks)
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "Revenue was $2B." in prompt
    assert "What was revenue?" in prompt


def test_build_prompt_includes_metadata():
    chunks = [
        Chunk(text="Some text.", company="NFLX", year=2024, section="Item 7"),
    ]
    prompt = build_prompt("query", chunks)
    assert "NFLX" in prompt
    assert "2024" in prompt
    assert "Item 7" in prompt


def test_extract_citations_finds_bracketed_numbers():
    text = "Revenue was $2B [1]. Risk is high [2]. Growth expected [1]."
    citations = extract_citations(text)
    assert citations == {1, 2}


def test_extract_citations_handles_no_citations():
    citations = extract_citations("No citations here.")
    assert citations == set()


def test_format_source_label():
    chunk = Chunk(text="x", company="NFLX", year=2024, section="Item 7")
    label = format_source_label(1, chunk)
    assert "NFLX" in label
    assert "2024" in label
    assert "Item 7" in label
    assert "[1]" in label
