"""Parse 10-K HTML filings into structured sections."""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup


# Standard 10-K item headers we care about
ITEM_PATTERNS = {
    "Item 1A": r"item\s+1a\.?\s*risk\s+factors",
    "Item 1": r"item\s+1\.?\s*business",
    "Item 7A": r"item\s+7a\.?\s*quantitative",
    "Item 7": r"item\s+7\.?\s*management",
    "Item 8": r"item\s+8\.?\s*financial\s+statements",
}


@dataclass
class Section:
    """A parsed section from a 10-K filing."""

    company: str
    year: int
    section: str
    section_name: str
    text: str


def parse_10k_html(
    html: str,
    company: str,
    year: int,
) -> list[Section]:
    """Parse 10-K HTML into a list of Section objects.

    Extracts major sections (Items 1, 1A, 7, 7A, 8) by finding
    section headers and capturing text until the next header.
    """
    soup = BeautifulSoup(html, "html.parser")
    full_text = soup.get_text(separator="\n")

    # Find all section start positions
    found: list[tuple[str, str, int]] = []
    for section_id, pattern in ITEM_PATTERNS.items():
        for match in re.finditer(pattern, full_text, re.IGNORECASE):
            section_name = _section_name(section_id)
            found.append((section_id, section_name, match.start()))

    if not found:
        return []

    # Sort by position in document
    found.sort(key=lambda x: x[2])

    # Extract text between consecutive headers
    sections: list[Section] = []
    for i, (sec_id, sec_name, start) in enumerate(found):
        # Find the start of actual content (after the header line)
        header_end = full_text.index("\n", start) + 1 if "\n" in full_text[start:] else start
        # End at next section or end of document
        end = found[i + 1][2] if i + 1 < len(found) else len(full_text)
        text = full_text[header_end:end].strip()
        text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excessive newlines

        if text:
            sections.append(
                Section(
                    company=company,
                    year=year,
                    section=sec_id,
                    section_name=sec_name,
                    text=text,
                )
            )

    return sections


_SECTION_NAMES = {
    "Item 1": "Business",
    "Item 1A": "Risk Factors",
    "Item 7": "Management's Discussion and Analysis",
    "Item 7A": "Market Risk Disclosures",
    "Item 8": "Financial Statements",
}


def _section_name(section_id: str) -> str:
    return _SECTION_NAMES.get(section_id, section_id)
