"""Shared test fixtures."""

import pytest


@pytest.fixture
def sample_10k_html():
    """Minimal 10-K HTML with known sections."""
    return """
    <html><body>
    <p style="font-weight:bold">ITEM 1. BUSINESS</p>
    <p>Acme Corp is a technology company that provides cloud services.
    The company was founded in 2010 and serves customers worldwide.
    Our primary products include data analytics and machine learning platforms.</p>

    <p style="font-weight:bold">ITEM 1A. RISK FACTORS</p>
    <p>We face significant competition from larger technology companies.
    Cybersecurity threats could disrupt our operations.
    Changes in regulations may impact our business model.</p>

    <p style="font-weight:bold">ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</p>
    <p>Revenue increased 15% to $2.3 billion in fiscal year 2024.
    Operating expenses grew 8% driven by R&amp;D investments.
    We expect continued growth in our cloud segment.</p>

    <p style="font-weight:bold">ITEM 8. FINANCIAL STATEMENTS</p>
    <p>Total revenue: $2,300,000,000. Net income: $450,000,000.
    Total assets: $8.1 billion. Long-term debt: $1.2 billion.</p>
    </body></html>
    """


@pytest.fixture
def sample_section():
    """A single parsed section dict."""
    return {
        "company": "ACME",
        "year": 2024,
        "section": "Item 1A",
        "section_name": "Risk Factors",
        "text": (
            "We face significant competition from larger technology companies. "
            "Cybersecurity threats could disrupt our operations. "
            "Changes in regulations may impact our business model."
        ),
    }
