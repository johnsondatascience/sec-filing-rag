"""LLM generation with citation enforcement."""

from __future__ import annotations

import re

from anthropic import Anthropic

from src.chunker import Chunk
from src.config import LLM_MODEL, MAX_CONTEXT_CHUNKS


SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer questions using ONLY the provided sources. "
    "For each claim, cite the source as [1], [2], etc. "
    "If the sources don't contain enough information, say so explicitly. "
    "Be precise and quantitative where possible."
)


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    """Build a prompt with numbered source references."""
    sources_block = "\n\n".join(
        f"[{i + 1}] {format_source_label(i + 1, c)}\n\"{c.text}\""
        for i, c in enumerate(chunks[:MAX_CONTEXT_CHUNKS])
    )
    return f"Sources:\n{sources_block}\n\nQuestion: {query}"


def format_source_label(number: int, chunk: Chunk) -> str:
    """Format a source citation label."""
    return f"[{number}] {chunk.company} 10-K {chunk.year}, {chunk.section}"


def extract_citations(text: str) -> set[int]:
    """Extract citation numbers from generated text."""
    return {int(m) for m in re.findall(r"\[(\d+)\]", text)}


def generate_answer(
    query: str,
    chunks: list[Chunk],
) -> dict:
    """Generate a cited answer using Claude.

    Returns dict with keys: answer, citations, sources.
    """
    client = Anthropic()
    prompt = build_prompt(query, chunks)

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    answer_text = response.content[0].text
    cited_nums = extract_citations(answer_text)

    # Build source references for cited chunks only
    sources = {}
    for i, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS]):
        num = i + 1
        if num in cited_nums:
            sources[num] = {
                "label": format_source_label(num, chunk),
                "text": chunk.text,
                "company": chunk.company,
                "year": chunk.year,
                "section": chunk.section,
            }

    return {
        "answer": answer_text,
        "citations": cited_nums,
        "sources": sources,
    }
