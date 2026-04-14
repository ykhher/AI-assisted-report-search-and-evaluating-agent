"""Filtering utilities for search results."""

from __future__ import annotations

from typing import Any


DEFAULT_KEYWORDS = [
    "market size",
    "forecast",
    "cagr",
    "revenue",
    "industry report",
]


def _score_value(result: dict[str, Any]) -> float:
    """Read a coarse score from a search result."""
    raw_score = result.get("score", result.get("relevance", 0.0))
    try:
        return float(raw_score)
    except (TypeError, ValueError):
        return 0.0


def filter_results(
    results: list[dict[str, Any]],
    min_score: float = 0.0,
    keywords: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Keep results that clear a score floor and match at least one keyword."""
    active_keywords = keywords or DEFAULT_KEYWORDS
    filtered: list[dict[str, Any]] = []

    for result in results:
        if _score_value(result) < min_score:
            continue

        text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
        if not active_keywords or any(keyword.lower() in text for keyword in active_keywords):
            filtered.append(result)

    return filtered
