"""Filtering utilities for search results."""

from __future__ import annotations

from typing import Any, Dict, List


def filter_results(
    results: List[Dict[str, Any]],
    min_score: float = 0.0,
    keywords: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Filter results by score and optionally by keyword matching.

    The keyword filter helps focus on results that look like market/forecast reports.
    """

    if keywords is None:
        keywords = [
            "market size",
            "forecast",
            "cagr",
            "revenue",
            "industry report",
        ]

    filtered: List[Dict[str, Any]] = []

    for result in results:
        score = result.get("score") or result.get("relevance") or 0
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        if score < min_score:
            continue

        title = (result.get("title") or "").strip()
        snippet = (result.get("snippet") or "").strip()
        text = f"{title} {snippet}".lower()

        if any(k in text for k in keywords):
            filtered.append(result)

    return filtered
