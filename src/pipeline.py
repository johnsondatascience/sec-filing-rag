"""End-to-end ingestion pipeline: download → parse → chunk → index."""

from __future__ import annotations

import json
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from src.chunker import Chunk, chunk_fixed, chunk_section_aware
from src.config import CHROMA_DIR, PROCESSED_DIR, RAW_DIR, TICKERS
from src.downloader import download_10k_filings
from src.indexer import build_bm25_index, build_chroma_index, save_chunks_metadata
from src.parser import parse_10k_html


def find_filing_html(filing_dir: Path) -> str | None:
    """Extract the 10-K HTML content from an EDGAR filing directory.

    sec-edgar-downloader saves a full-submission.txt containing embedded
    SGML documents. We extract the first 10-K HTML document from it.
    Returns the HTML content as a string, or None if not found.
    """
    # Try standalone HTML files first
    htm_files = list(filing_dir.rglob("*.htm*"))
    if htm_files:
        largest = max(htm_files, key=lambda p: p.stat().st_size)
        return largest.read_text(encoding="utf-8", errors="replace")

    # Fall back to extracting from full-submission.txt
    submission = filing_dir / "full-submission.txt"
    if not submission.exists():
        return None

    content = submission.read_text(encoding="utf-8", errors="replace")

    # Find the 10-K document within the SGML submission
    # Documents are delimited by <DOCUMENT> ... </DOCUMENT> tags
    doc_pattern = re.compile(
        r"<DOCUMENT>\s*<TYPE>10-K\s.*?<TEXT>(.*?)</TEXT>",
        re.DOTALL | re.IGNORECASE,
    )
    match = doc_pattern.search(content)
    if match:
        return match.group(1)

    return None


def ingest(
    tickers: list[str] | None = None,
    filing_limit: int = 2,
    strategy: str = "fixed",
) -> list[Chunk]:
    """Full ingestion pipeline.

    1. Download 10-K filings from EDGAR
    2. Parse HTML into sections
    3. Chunk sections using the specified strategy
    4. Return all chunks
    """
    tickers = tickers or TICKERS
    all_chunks: list[Chunk] = []

    # Step 1: Download
    print(f"Downloading 10-K filings for {len(tickers)} tickers...")
    filings = download_10k_filings(tickers=tickers, limit=filing_limit)

    # Step 2-3: Parse and chunk each filing
    for ticker, filing_paths in filings.items():
        for fpath in filing_paths:
            filing_dir = Path(fpath)
            html = find_filing_html(filing_dir)
            if html is None:
                print(f"  SKIP {ticker} {filing_dir.name}: no HTML found")
                continue

            # Extract year from directory name (accession number contains date info)
            year = _extract_year(filing_dir.name)

            sections = parse_10k_html(html, company=ticker, year=year)
            print(f"  {ticker} {year}: {len(sections)} sections parsed")

            for section in sections:
                if strategy == "fixed":
                    chunks = chunk_fixed(
                        section.text, company=ticker, year=year, section=section.section
                    )
                elif strategy == "section_aware":
                    chunks = chunk_section_aware(
                        section.text, company=ticker, year=year, section=section.section
                    )
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def _extract_year(dirname: str) -> int:
    """Best-effort year extraction from EDGAR accession number directory name."""
    # Accession numbers look like 0001193125-24-123456 (the 24 = 2024)
    parts = dirname.split("-")
    for part in parts:
        if len(part) == 2 and part.isdigit():
            yr = int(part)
            return 2000 + yr if yr < 50 else 1900 + yr
    return 0


def build_indexes(
    chunks: list[Chunk],
) -> tuple[chromadb.Collection, BM25Okapi, list[Chunk]]:
    """Build both vector and BM25 indexes from chunks."""
    print("Building ChromaDB index...")
    collection = build_chroma_index(chunks)
    print(f"  ChromaDB: {collection.count()} vectors indexed")

    print("Building BM25 index...")
    bm25, aligned = build_bm25_index(chunks)
    print(f"  BM25: {bm25.corpus_size} documents indexed")

    # Save metadata for reproducibility
    save_chunks_metadata(chunks, PROCESSED_DIR / "chunks_metadata.json")

    return collection, bm25, aligned


if __name__ == "__main__":
    # Run full pipeline with a small test
    chunks = ingest(tickers=["NFLX"], filing_limit=1, strategy="fixed")
    if chunks:
        collection, bm25, aligned = build_indexes(chunks)
        print("Pipeline complete!")
