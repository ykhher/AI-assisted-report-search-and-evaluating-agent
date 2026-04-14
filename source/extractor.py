"""Phase 3 signal extraction for the report credibility ranking system."""

from __future__ import annotations

import re
from typing import Any

from local_qwen import assess_text_signals

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
_INLINE_FOOTNOTE_PATTERN = re.compile(
    r"(?:(?<=[A-Za-z\)\].,;:\[])\s*(?P<num>(?:[1-9]|[12]\d|30))(?=(?:\s|$|[)\].,;:]))|(?P<sup>[⁰¹²³⁴⁵⁶⁷⁸⁹]+))"
)
_BOTTOM_REFERENCE_LINE_PATTERN = re.compile(
    r"^\s*(?P<num>(?:[1-9]|[12]\d|30))[.)]?\s+(?P<line>.+)$",
    re.MULTILINE,
)
_REFERENCE_HINT_PATTERN = re.compile(
    r"\b(?:source|report|analysis|outlook|study|dataset|data)\b",
    re.IGNORECASE,
)
_URL_OR_YEAR_PATTERN = re.compile(r"(?:https?://|www\.|\b(?:19|20)\d{2}\b)", re.IGNORECASE)
_INSTITUTION_PATTERNS = (
    re.compile(r"\biea\b", re.IGNORECASE),
    re.compile(r"\bipcc\b", re.IGNORECASE),
    re.compile(r"(?:\bun\b|\bunited nations\b)", re.IGNORECASE),
    re.compile(r"\bworld bank\b", re.IGNORECASE),
    re.compile(r"\bmckinsey\b", re.IGNORECASE),
    re.compile(r"\boxford economics\b", re.IGNORECASE),
)
_SUPERSCRIPT_TRANSLATION = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
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
_SOURCE_NAME_SCORES = {
    "world bank": 0.96,
    "oecd": 0.94,
    "iea": 0.92,
    "ipcc": 0.92,
    "united nations": 0.90,
    "un": 0.88,
    "stanford": 0.88,
    "mckinsey": 0.84,
    "oxford economics": 0.84,
    "world economic forum": 0.82,
    "bcg": 0.80,
    "boston consulting group": 0.80,
    "deloitte": 0.78,
}


def clean_text(text: str) -> str:
    """Lowercase text and collapse extra whitespace."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _metadata_dict(metadata: Any) -> dict[str, Any]:
    """Return metadata as a plain dictionary when possible."""
    return metadata if isinstance(metadata, dict) else {}


def has_methodology(text: str) -> int:
    """Return 1 only when at least two strong methodology signals appear."""
    cleaned = clean_text(text)
    matches = sum(1 for keyword in _METHODOLOGY_KEYWORDS if keyword in cleaned)
    return int(matches >= 2)


def footnote_score(text: str) -> float:
    """Detect inline footnote-style citations such as `trend acceleration.1` or superscripts like `¹`."""
    raw_text = str(text)
    count = 0

    for match in _INLINE_FOOTNOTE_PATTERN.finditer(raw_text):
        candidate = (match.group("num") or match.group("sup") or "").translate(_SUPERSCRIPT_TRANSLATION)
        if candidate.isdigit() and 1 <= int(candidate) <= 30:
            count += 1

    return round(min(count / 30, 1.0), 3)


def bottom_reference_score(text: str) -> float:
    """Detect numbered bottom-of-page source notes without requiring a `References` heading."""
    raw_text = str(text)
    count = 0

    for match in _BOTTOM_REFERENCE_LINE_PATTERN.finditer(raw_text):
        line = match.group("line").strip()
        if not line:
            continue

        count += 1
        if len(re.findall(r"\b\w+\b", line)) >= 3:
            count += 1
        if _REFERENCE_HINT_PATTERN.search(line):
            count += 1
        count += sum(1 for pattern in _INSTITUTION_PATTERNS if pattern.search(line))
        if _URL_OR_YEAR_PATTERN.search(line):
            count += 1

    return round(min(count / 15, 1.0), 3)


def institution_score(text: str) -> float:
    """Measure how strongly the text cites authoritative institutions often seen in consulting reports."""
    cleaned = clean_text(text)
    total_matches = 0
    unique_hits = 0

    for pattern in _INSTITUTION_PATTERNS:
        matches = pattern.findall(cleaned)
        if matches:
            total_matches += len(matches)
            unique_hits += 1

    count = total_matches + unique_hits
    return round(min(count / 10, 1.0), 3)


def in_text_citation_score(text: str) -> float:
    """Backward-compatible wrapper for inline citation detection."""
    return footnote_score(text)


def reference_section_score(text: str) -> float:
    """Backward-compatible wrapper for bottom reference detection."""
    return bottom_reference_score(text)


def numbered_reference_score(text: str) -> float:
    """Backward-compatible wrapper for numbered reference detection."""
    return bottom_reference_score(text)


def compute_citation_score(text: str) -> float:
    """Combine consulting-style citation signals into one robust credibility feature."""
    citation_score = max(bottom_reference_score(text), footnote_score(text))
    citation_score += 0.2 * institution_score(text)
    return round(min(citation_score, 1.0), 3)


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


def source_score(source: Any, context_text: str = "") -> float:
    """Map a source domain or publisher name to a normalized credibility prior, with optional LLM support."""
    if isinstance(source, (int, float)):
        return round(max(0.0, min(float(source), 1.0)), 3)

    source_text = str(source).lower().strip()
    if not source_text:
        return 0.0

    heuristic_score = 0.5
    for suffix, score in _SOURCE_REPUTATION_MAP.items():
        if source_text.endswith(suffix) or suffix in source_text:
            heuristic_score = max(heuristic_score, score)

    for name, score in _SOURCE_NAME_SCORES.items():
        if name in source_text:
            heuristic_score = max(heuristic_score, score)

    llm_scores = assess_text_signals(str(context_text or ""), source=source_text)
    llm_source_score = float(llm_scores.get("source_score", 0.0))

    return round(max(heuristic_score, llm_source_score), 3)


def extract_signals(text: str, metadata: dict) -> dict:
    """Extract normalized credibility signals from real-world report text."""
    raw_text = str(text)
    metadata_map = _metadata_dict(metadata)
    year = metadata_map.get("year")

    footnotes = footnote_score(raw_text)
    bottom_references = bottom_reference_score(raw_text)
    institutions = institution_score(raw_text)
    source_name = str(metadata_map.get("source", "")).strip()
    llm_scores = assess_text_signals(raw_text, source=str(source_name))

    methodology = round(max(float(has_methodology(raw_text)), float(llm_scores.get("methodology_score", 0.0))), 3)
    citation = round(max(compute_citation_score(raw_text), float(llm_scores.get("reference_score", 0.0))), 3)
    consistency = round(float(llm_scores.get("consistency_score", 0.0)), 3)
    source_value = round(source_score(source_name, raw_text), 3) if source_name else 0.0

    return {
        "methodology": methodology,
        "footnote_score": footnotes,
        "bottom_reference_score": bottom_references,
        "institution_score": institutions,
        "llm_reference_score": round(float(llm_scores.get("reference_score", 0.0)), 3),
        "llm_methodology_score": round(float(llm_scores.get("methodology_score", 0.0)), 3),
        "llm_consistency_score": consistency,
        "llm_source_score": round(float(llm_scores.get("source_score", 0.0)), 3),
        "llm_reason": str(llm_scores.get("reason", "")).strip(),
        "citation_score": citation,
        "consistency_score": consistency,
        "source": source_value,
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
