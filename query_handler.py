"""Query expansion helpers for report discovery."""

from __future__ import annotations

from local_qwen import suggest_search_queries


def generate_queries(user_input: str) -> list[str]:
    """Generate search-friendly query variants and enrich them with local Qwen when available."""
    topic = " ".join(str(user_input or "").strip().lower().split())
    if not topic:
        return []

    market_keywords = ["market", "forecast", "industry", "outlook", "cagr", "report"]
    if not any(keyword in topic for keyword in market_keywords):
        topic = f"{topic} market"

    templates = [
        f"{topic} size forecast report",
        f"{topic} industry outlook CAGR projection",
        f"{topic} market research report filetype:pdf",
        f"{topic} revenue forecast analysis",
    ]

    llm_queries = suggest_search_queries(topic, max_queries=3)

    unique_queries: list[str] = []
    seen: set[str] = set()
    for query in [*templates, *llm_queries]:
        normalized = query.lower().strip()
        if normalized and normalized not in seen:
            unique_queries.append(query)
            seen.add(normalized)

    return unique_queries
