"""Phase 3 signal extraction for the report credibility ranking system."""

from __future__ import annotations

import json
import re
from typing import Any

_CURRENT_YEAR = 2026
_METHODOLOGY_KEYWORDS = (
    "methodology",
    "methods",
    "research design",
    "data collection",
    "sampling",
    "we conducted",
    "we analyzed",
)
_NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
_CITATION_BRACKET_PATTERN = re.compile(r"\[\d+\]")
_CITATION_YEAR_PATTERN = re.compile(r"\((?:19|20)\d{2}\)")
_NUMBERED_REFERENCE_PATTERN = re.compile(r"(?:^|\n)\s*\d+\.\s", re.MULTILINE)
_REFERENCE_SECTION_KEYWORDS = ("references", "bibliography", "sources")
_SECTION_NAMES = (
    "introduction",
    "methodology",
    "methods",
    "results",
    "findings",
    "discussion",
    "conclusion",
    "references",
)

# Section header patterns found in formal reports
_SECTION_PATTERNS = re.compile(
    r"\b(introduction|methodology|methods|results|findings|discussion|"
    r"conclusion|references|bibliography|executive summary|appendix)\b",
    re.IGNORECASE,
)

# Structural keywords that distinguish reports from blog posts
_REPORT_KEYWORDS = re.compile(
    r"\b(methodology|methods|research design|data collection|sampling|"
    r"bibliography|references|findings|objectives|scope|limitations|recommendations)\b",
    re.IGNORECASE,
)

_SOURCE_REPUTATION_MAP = {
    ".gov": 0.95,
    ".edu": 0.90,
    ".org": 0.80,
    ".net": 0.65,
    ".com": 0.55,
    ".io": 0.45,
}


def clean_text(text: str) -> str:
    """Lowercase text and collapse extra whitespace."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def has_methodology(text: str) -> int:
    """Return 1 only when at least two strong methodology signals appear."""
    cleaned = clean_text(text)
    matches = sum(1 for keyword in _METHODOLOGY_KEYWORDS if keyword in cleaned)
    return int(matches >= 2)


def in_text_citation_score(text: str) -> float:
    """Detect academic-style in-text citations like `[1]` and `(2024)`."""
    raw_text = str(text)
    count = (
        len(_CITATION_BRACKET_PATTERN.findall(raw_text))
        + len(_CITATION_YEAR_PATTERN.findall(raw_text))
    )
    return round(min(count / 20, 1.0), 3)


def reference_section_score(text: str) -> float:
    """Detect a reference section and count lines that look like references."""
    raw_text = str(text)
    lowered = raw_text.lower()
    section_positions = [lowered.rfind(keyword) for keyword in _REFERENCE_SECTION_KEYWORDS if keyword in lowered]
    if not section_positions:
        return 0.0

    section_start = max(section_positions)
    section_text = raw_text[section_start:]
    lines = [line.strip() for line in section_text.splitlines() if line.strip()]
    if len(lines) <= 1:
        lines = [segment.strip() for segment in re.split(r"(?<=[.;])\s+", section_text) if segment.strip()]

    reference_lines = 0
    for line in lines[1:]:
        word_count = len(re.findall(r"\b\w+\b", line))
        if "http" in line.lower() or "www" in line.lower() or word_count > 8:
            reference_lines += 1

    return round(min(reference_lines / 20, 1.0), 3)


def numbered_reference_score(text: str) -> float:
    """Detect numbered reference lines like `1. McKinsey (2023)`."""
    count = len(_NUMBERED_REFERENCE_PATTERN.findall(str(text)))
    return round(min(count / 15, 1.0), 3)


def compute_citation_score(text: str) -> float:
    """Combine academic and industry-style reference evidence into one citation score."""
    base_score = max(in_text_citation_score(text), reference_section_score(text))
    boosted_score = base_score + 0.3 * numbered_reference_score(text)
    return round(min(boosted_score, 1.0), 3)


def _is_year_token(token: str) -> bool:
    """Return True when a numeric token looks like a calendar year."""
    try:
        value = int(token.rstrip("%"))
    except ValueError:
        return False
    return 1900 <= value <= 2100


def data_density(text: str) -> float:
    """Compute normalized numeric density while excluding likely year values."""
    cleaned = clean_text(text)
    words = cleaned.split()
    num_words = len(words)
    if num_words == 0:
        return 0.0

    numeric_tokens = [token for token in _NUMBER_PATTERN.findall(cleaned) if not _is_year_token(token)]
    raw_density = len(numeric_tokens) / num_words
    return round(min(raw_density / 0.02, 1.0), 3)


def length_score(text: str) -> float:
    """Normalize document length by a 10,000-word cap."""
    cleaned = clean_text(text)
    num_words = len(cleaned.split())
    return round(min(num_words / 10000, 1.0), 3)


def recency_score(year: int) -> float:
    """Compute recency using the fixed current year 2026."""
    try:
        numeric_year = int(year)
    except (TypeError, ValueError):
        return 0.0

    score = 1 - (_CURRENT_YEAR - numeric_year) / 5
    return round(max(0.0, min(score, 1.0)), 3)


def compute_structure_score(text: str) -> float:
    """Measure how many standard report sections are present."""
    cleaned = clean_text(text)
    detected = sum(1 for section in _SECTION_NAMES if section in cleaned)
    return round(min(detected / len(_SECTION_NAMES), 1.0), 3)


def source_score(source: Any) -> float:
    """Map a source domain or numeric value to a normalized credibility prior."""
    if isinstance(source, (int, float)):
        return round(max(0.0, min(float(source), 1.0)), 3)

    source_text = str(source).lower().strip()
    if not source_text:
        return 0.0

    for suffix, score in _SOURCE_REPUTATION_MAP.items():
        if source_text.endswith(suffix) or suffix in source_text:
            return score

    return 0.5


def extract_signals(text: str, metadata: dict) -> dict:
    """Extract the required normalized Phase 3 signals."""
    raw_text = str(text)
    year = metadata.get("year") if isinstance(metadata, dict) else None

    return {
        "methodology": has_methodology(raw_text),
        "citation_score": compute_citation_score(raw_text),
        "data_density": data_density(raw_text),
        "length": length_score(raw_text),
        "recency": recency_score(year),
        "structure_score": compute_structure_score(raw_text),
    }


def is_report(text: str, metadata: dict) -> bool:
    """Heuristic gate for filtering search results to report-like documents."""
    cleaned = clean_text(text)
    word_count = len(cleaned.split())
    section_matches = set(_SECTION_PATTERNS.findall(cleaned))
    report_kw_matches = len(_REPORT_KEYWORDS.findall(cleaned))

    return (
        word_count >= 1500
        and len(section_matches) >= 3
        and report_kw_matches >= 2
    )


if __name__ == "__main__":
    test_cases = [
        {
            "name": "Academic-style text",
            "text": (
                "Introduction. Methodology: we conducted a longitudinal study with data collection across 250 firms. "
                "Results show 18% growth [1] [2] and projected gains in 2025 (2024) (2023). "
                "Conclusion summarizes the findings and limitations."
            ),
            "metadata": {"year": 2024},
        },
        {
            "name": "Industry report with references section",
            "text": (
                "Introduction\nMethodology\nWe analyzed survey data from 180 executives.\nResults\nRevenue is expected to increase by 12%.\n"
                "Conclusion\nThe outlook remains positive.\nReferences\n"
                "1. McKinsey (2023) Global AI Report https://www.mckinsey.com/report\n"
                "2. Deloitte report www.deloitte.com/insights/future-of-ai\n"
                "3. World Bank industry outlook and source notes for 2024 and 2025\n"
            ),
            "metadata": {"year": 2024},
        },
        {
            "name": "Blog-like text",
            "text": (
                "This blog talks about cool trends and opinions. It has no methodology, no references, and no source list."
            ),
            "metadata": {"year": 2024},
        },
    ]

    for case in test_cases:
        text = case["text"]
        print(f"\n--- {case['name']} ---")
        print(json.dumps({
            "in_text_citation_score": in_text_citation_score(text),
            "reference_section_score": reference_section_score(text),
            "numbered_reference_score": numbered_reference_score(text),
            "citation_score": compute_citation_score(text),
            "signals": extract_signals(text, case["metadata"]),
        }, indent=2))
