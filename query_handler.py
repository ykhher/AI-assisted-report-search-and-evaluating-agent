"""Query expansion helpers for report discovery."""

from __future__ import annotations


def generate_queries(user_input: str) -> list[str]:
    """Generate a small set of search-friendly query variants for the same topic."""
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

    unique_queries: list[str] = []
    seen: set[str] = set()
    for query in templates:
        if query not in seen:
            unique_queries.append(query)
            seen.add(query)

    return unique_queries
