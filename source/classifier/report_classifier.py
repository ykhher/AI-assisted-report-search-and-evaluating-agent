"""Classify document type and estimate report validity."""

from __future__ import annotations

import re
from typing import Any, Optional


REPORT_KEYWORDS = {
    "report", "analysis", "analytical", "assessment", "evaluation",
    "findings", "conclusion", "conclusions", "results", "outcome",
    "performance review", "annual report", "quarterly report", "status report",
}

WHITEPAPER_KEYWORDS = {
    "whitepaper", "white paper", "white-paper", "position paper",
    "technical paper", "research paper", "conceptual framework",
}

BENCHMARK_KEYWORDS = {
    "benchmark", "benchmarking", "comparison", "competitive analysis",
    "market sizing", "performance comparison", "study comparison",
}

SURVEY_KEYWORDS = {
    "survey", "questionnaire", "empirical study", "research study",
    "statistical analysis", "data analysis", "field study",
}

RESEARCH_NOTE_KEYWORDS = {
    "research note", "technical note", "note", "brief", "memo",
    "update", "insight", "technical brief",
}

DECK_KEYWORDS = {
    "presentation", "slides", "slide deck", "deck", "ppt",
    "powerpoint", "keynote", "presentation slides",
}

BROCHURE_KEYWORDS = {
    "brochure", "guide", "handbook", "manual", "directory",
    "reference guide", "field guide", "how-to", "tutorial",
}

BLOG_KEYWORDS = {
    "blog", "blog post", "post", "opinion", "commentary", "column",
    "thought leadership", "perspective", "editorial", "my thoughts",
}

LANDING_PAGE_KEYWORDS = {
    "landing page", "product page", "pricing page", "about us",
    "welcome page", "home page",
}

PROMOTIONAL_KEYWORDS = {
    "sign up", "free trial", "subscribe", "subscribe now",
    "buy now", "limited offer", "exclusive", "get started",
    "learn more", "contact us", "demo", "request demo",
    "schedule demo", "request trial",
}

SECTION_MARKERS = {
    r"^##\s+",
    r"^###\s+",
    r"^####\s+",
    r"^(Section|Chapter|Part)\s+\d+",
    r"^\d+\.\s+[A-Z]",
    r"^\s*-\s+[A-Z][a-z]+:",
}

REPORT_METHODOLOGY_KEYWORDS = {
    "methodology", "method", "approach", "we conducted", "we analyzed",
    "research design", "framework", "evaluation criteria", "data collection",
    "experimental design", "sample size",
}

REFERENCE_PATTERNS = (
    r"\[(\d+)\]",
    r"(\w+\s+et al\.?\s+\d{4})",
    r"(references|bibliography|citations)\s*:",
)

STRUCTURE_SECTION_KEYWORDS = {
    "executive summary", "introduction", "background", "methodology",
    "results", "findings", "analysis", "conclusion", "conclusions",
    "references", "appendix", "limitations", "discussion",
}


def _normalize_text(text: str) -> str:
    """Normalize text to lowercase for keyword matching."""
    return text.lower().strip() if text else ""


def _match_keywords(text: str, keyword_set: set[str]) -> bool:
    """Check if any keyword from a set appears in normalized text."""
    normalized = _normalize_text(text)
    return any(keyword in normalized for keyword in keyword_set)


def _build_result(report_type: str, validity: float) -> dict[str, Any]:
    """Build a normalized classifier output."""
    return {
        "report_type": report_type,
        "report_validity_score": min(validity, 1.0),
    }


def _count_section_markers(text: str) -> int:
    """Count approximate number of sections or subsections detected."""
    if not text:
        return 0
    return sum(len(re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)) for pattern in SECTION_MARKERS)


def _count_references(text: str) -> int:
    """Count approximate number of reference-like patterns."""
    if not text:
        return 0
    return sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in REFERENCE_PATTERNS)


def _contains_report_structure(text: str) -> float:
    """Return a heuristic report-structure score in the range [0, 1]."""
    if not text:
        return 0.0

    normalized = _normalize_text(text)
    structure_keyword_score = min(
        sum(1 for kw in STRUCTURE_SECTION_KEYWORDS if kw in normalized) / len(STRUCTURE_SECTION_KEYWORDS),
        1.0,
    )
    section_score = min(_count_section_markers(text) / 10, 1.0)
    reference_score = min(_count_references(text) / 5, 1.0)
    return (structure_keyword_score * 0.4) + (section_score * 0.4) + (reference_score * 0.2)


def _contains_methodology_language(text: str) -> bool:
    """Check if text contains research/report methodology indicators."""
    return _match_keywords(text, REPORT_METHODOLOGY_KEYWORDS)


def _is_promotional(title: str, text: str) -> bool:
    """Detect promotional or marketing language."""
    combined = f"{_normalize_text(title)} {_normalize_text(text[:500])}"
    return _match_keywords(combined, PROMOTIONAL_KEYWORDS)


def _is_pdf(metadata: Optional[dict]) -> bool:
    """Check if document metadata suggests PDF format."""
    if not metadata:
        return False
    fmt = str(metadata.get("format", "")).lower()
    file_type = str(metadata.get("file_type", "")).lower()
    return "pdf" in fmt or "pdf" in file_type


def classify_report_type(
    title: str = "",
    text: str = "",
    metadata: dict | None = None,
) -> dict[str, Any]:
    """Classify document type and return a report-validity score."""
    normalized_title = _normalize_text(title)
    normalized_text = _normalize_text(text)

    if _is_promotional(title, text):
        validity = 0.3 if _match_keywords(normalized_text, LANDING_PAGE_KEYWORDS) else 0.2
        return _build_result("landing_page", validity)

    if _match_keywords(normalized_title, WHITEPAPER_KEYWORDS):
        return _build_result("whitepaper", 0.6 + (_contains_report_structure(text) * 0.3))

    if _match_keywords(normalized_title, BENCHMARK_KEYWORDS):
        has_methodology = _contains_methodology_language(text)
        return _build_result("benchmark", 0.65 + (0.2 if has_methodology else 0.0))

    if _match_keywords(normalized_title, DECK_KEYWORDS):
        return _build_result("deck", 0.7 if _is_pdf(metadata) else 0.6)

    if _match_keywords(normalized_title, BROCHURE_KEYWORDS):
        return _build_result("brochure", 0.5)

    if _match_keywords(normalized_title, BLOG_KEYWORDS):
        return _build_result("blog", 0.4)

    if _match_keywords(normalized_title, RESEARCH_NOTE_KEYWORDS):
        has_methodology = _contains_methodology_language(text)
        return _build_result("research_note", 0.55 + (0.2 if has_methodology else 0.05))

    if _match_keywords(normalized_title, SURVEY_KEYWORDS):
        has_methodology = _contains_methodology_language(text)
        return _build_result("survey", 0.65 + (0.2 if has_methodology else 0.05))

    if _match_keywords(normalized_title, REPORT_KEYWORDS):
        return _build_result("report", 0.70 + (_contains_report_structure(text) * 0.25))

    structure_score = _contains_report_structure(text)
    has_methodology = _contains_methodology_language(text)

    if structure_score > 0.6 and has_methodology:
        return _build_result("report", 0.65 + (structure_score * 0.25))

    if structure_score > 0.4 and not has_methodology:
        if _is_pdf(metadata):
            return _build_result("whitepaper", 0.55 + (structure_score * 0.20))
        return _build_result("research_note", 0.50 + (structure_score * 0.15))

    text_head = text[:500] if text else ""
    if text_head and _match_keywords(text_head, {"blog post", "blog", "article", "my thoughts", "opinion"}):
        return _build_result("blog", 0.45)

    return _build_result("unknown", min(structure_score * 0.4, 0.5))
