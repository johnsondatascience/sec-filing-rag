"""Chunking strategies for 10-K filing sections."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A text chunk with source metadata."""

    text: str
    company: str
    year: int
    section: str
    chunk_id: str = field(default="")
    strategy: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            raw = f"{self.company}-{self.year}-{self.section}-{hash(self.text)}-{id(self)}"
            self.chunk_id = hashlib.md5(raw.encode()).hexdigest()[:12]


def chunk_fixed(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    *,
    company: str,
    year: int,
    section: str,
) -> list[Chunk]:
    """Split text into fixed-size token chunks with overlap.

    Uses whitespace tokenization (word count) as a fast proxy for token count.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [Chunk(text=text, company=company, year=year, section=section, strategy="fixed")]

    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(
            Chunk(text=chunk_text, company=company, year=year, section=section, strategy="fixed")
        )
        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks


def chunk_section_aware(
    text: str,
    chunk_size: int = 512,
    *,
    company: str,
    year: int,
    section: str,
) -> list[Chunk]:
    """Split by paragraph boundaries, merging small paragraphs up to chunk_size.

    Respects document structure by never splitting mid-paragraph unless
    a single paragraph exceeds chunk_size.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current_words: list[str] = []

    for para in paragraphs:
        para_words = para.split()

        # If adding this paragraph exceeds chunk_size, flush current
        if current_words and len(current_words) + len(para_words) > chunk_size:
            chunks.append(
                Chunk(
                    text=" ".join(current_words),
                    company=company,
                    year=year,
                    section=section,
                    strategy="section_aware",
                )
            )
            current_words = []

        # If a single paragraph exceeds chunk_size, use fixed chunking on it
        if len(para_words) > chunk_size:
            if current_words:
                chunks.append(
                    Chunk(
                        text=" ".join(current_words),
                        company=company,
                        year=year,
                        section=section,
                        strategy="section_aware",
                    )
                )
                current_words = []
            sub_chunks = chunk_fixed(
                para, chunk_size=chunk_size, overlap=50, company=company, year=year, section=section
            )
            for sc in sub_chunks:
                sc.strategy = "section_aware"
            chunks.extend(sub_chunks)
        else:
            current_words.extend(para_words)

    # Flush remaining
    if current_words:
        chunks.append(
            Chunk(
                text=" ".join(current_words),
                company=company,
                year=year,
                section=section,
                strategy="section_aware",
            )
        )

    return chunks
