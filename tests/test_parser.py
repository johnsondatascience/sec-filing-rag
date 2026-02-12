"""Tests for 10-K HTML parser."""

from src.parser import parse_10k_html, Section


def test_parse_extracts_known_sections(sample_10k_html):
    sections = parse_10k_html(sample_10k_html, company="ACME", year=2024)
    section_names = [s.section for s in sections]
    assert "Item 1" in section_names
    assert "Item 1A" in section_names
    assert "Item 7" in section_names


def test_parse_returns_section_objects(sample_10k_html):
    sections = parse_10k_html(sample_10k_html, company="ACME", year=2024)
    for s in sections:
        assert isinstance(s, Section)
        assert s.company == "ACME"
        assert s.year == 2024
        assert len(s.text) > 0


def test_parse_preserves_text_content(sample_10k_html):
    sections = parse_10k_html(sample_10k_html, company="ACME", year=2024)
    risk_section = next(s for s in sections if s.section == "Item 1A")
    assert "competition" in risk_section.text
    assert "Cybersecurity" in risk_section.text


def test_parse_empty_html():
    sections = parse_10k_html("<html><body></body></html>", company="X", year=2024)
    assert sections == []
