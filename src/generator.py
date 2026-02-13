"""LLM generation with citation enforcement."""

from __future__ import annotations

import os
import re

from openai import OpenAI

from src.chunker import Chunk
from src.config import MAX_CONTEXT_CHUNKS


# LM Studio defaults
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3")

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
    """Generate a cited answer using a local LLM via OpenAI-compatible API.

    Returns dict with keys: answer, citations, sources.
    """
    client = OpenAI(base_url=LLM_BASE_URL, api_key="lm-studio")
    prompt = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    answer_text = response.choices[0].message.content
    # Strip Qwen3 <think>...</think> reasoning blocks
    answer_text = re.sub(r"<think>.*?</think>\s*", "", answer_text, flags=re.DOTALL)
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
