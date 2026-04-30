"""Query expansion helpers for report discovery."""

from __future__ import annotations

import re
from datetime import datetime

from local_qwen import suggest_search_queries


REPORT_INTENT_KEYWORDS = {
    "analysis",
    "benchmark",
    "forecast",
    "industry",
    "market",
    "outlook",
    "report",
    "research",
    "study",
    "survey",
    "whitepaper",
}
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "best",
    "credible",
    "find",
    "for",
    "give",
    "latest",
    "me",
    "most",
    "recent",
    "show",
    "the",
    "top",
}
QUERY_TEMPLATES = [
    "{topic} {year_terms} report",
    "{topic} {year_terms} benchmark report",
    "{topic} {year_terms} market outlook",
    "{topic} {year_terms} industry analysis",
    "{topic} {year_terms} research report filetype:pdf",
    "{topic} survey methodology findings {year_terms}",
]


def _normalize_query_text(user_input: str) -> str:
    """Normalize free-text user input for query generation."""
    return " ".join(str(user_input or "").strip().lower().split())


def _extract_years(user_input: str) -> list[int]:
    """Return explicit years in first-seen order."""
    years: list[int] = []
    seen: set[int] = set()
    for match in re.findall(r"\b(?:19|20)\d{2}\b", str(user_input)):
        year = int(match)
        if year not in seen:
            years.append(year)
            seen.add(year)
    return years


def _year_terms(user_input: str) -> str:
    """Build a compact recency suffix for expanded search queries."""
    explicit_years = _extract_years(user_input)
    if explicit_years:
        return " ".join(str(year) for year in explicit_years[:2])

    current_year = datetime.now().year
    return f"{current_year} {current_year - 1}"


def _base_topic(user_input: str) -> str:
    """Build a focused topic string without throwing away domain words."""
    topic = _normalize_query_text(user_input)
    if not topic:
        return ""

    tokens = re.findall(r"[a-z0-9]+", topic)
    filtered_tokens = [
        token
        for token in tokens
        if token not in QUERY_STOPWORDS
        and token not in REPORT_INTENT_KEYWORDS
        and not re.fullmatch(r"(?:19|20)\d{2}", token)
    ]

    focused = " ".join(filtered_tokens).strip()
    return focused or topic


def _dedupe_queries(queries: list[str]) -> list[str]:
    """Keep unique non-empty queries in first-seen order."""
    unique_queries: list[str] = []
    seen: set[str] = set()

    for query in queries:
        normalized = _normalize_query_text(query)
        if not normalized or normalized in seen:
            continue
        unique_queries.append(query.strip())
        seen.add(normalized)

    return unique_queries


def generate_queries(
    user_input: str,
    *,
    topic: str | None = None,
    year_terms: str | None = None,
) -> list[str]:
    """Generate search-friendly query variants, with optional local Qwen help."""
    if topic is None:
        topic = _base_topic(user_input)
    if not topic:
        return []
    if year_terms is None:
        year_terms = _year_terms(user_input)

    normalized_input = _normalize_query_text(user_input)
    template_queries = [
        template.format(topic=topic, year_terms=year_terms)
        for template in QUERY_TEMPLATES
    ]

    try:
        llm_queries = suggest_search_queries(topic, max_queries=3)
    except Exception:
        llm_queries = []

    return _dedupe_queries([normalized_input, *template_queries, *llm_queries])
