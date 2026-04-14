"""Query expansion helpers for report discovery."""

from __future__ import annotations

from local_qwen import suggest_search_queries


MARKET_KEYWORDS = ["market", "forecast", "industry", "outlook", "cagr", "report"]
QUERY_TEMPLATES = [
    "{topic} size forecast report",
    "{topic} industry outlook CAGR projection",
    "{topic} market research report filetype:pdf",
    "{topic} revenue forecast analysis",
]


def _normalize_query_text(user_input: str) -> str:
    """Normalize free-text user input for query generation."""
    return " ".join(str(user_input or "").strip().lower().split())


def _base_topic(user_input: str) -> str:
    """Build a topic string that is broad enough for report-style search."""
    topic = _normalize_query_text(user_input)
    if not topic:
        return ""
    if any(keyword in topic for keyword in MARKET_KEYWORDS):
        return topic
    return f"{topic} market"


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


def generate_queries(user_input: str) -> list[str]:
    """Generate search-friendly query variants, with optional local Qwen help."""
    topic = _base_topic(user_input)
    if not topic:
        return []

    template_queries = [template.format(topic=topic) for template in QUERY_TEMPLATES]

    try:
        llm_queries = suggest_search_queries(topic, max_queries=3)
    except Exception:
        llm_queries = []

    return _dedupe_queries([*template_queries, *llm_queries])
