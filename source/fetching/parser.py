"""Helpers for parsing raw search API responses into a normalized list."""

from __future__ import annotations

import re
from typing import Any


_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


def _extract_hits(raw_data: Any) -> list[dict[str, Any]]:
    """Extract a list of result dictionaries from common API response shapes."""
    if isinstance(raw_data, list):
        return [item for item in raw_data if isinstance(item, dict)]
    if not isinstance(raw_data, dict):
        return []

    hits = (
        raw_data.get("organic_results")
        or raw_data.get("hits")
        or raw_data.get("results")
        or raw_data.get("documents")
        or raw_data.get("data")
        or raw_data.get("webPages", {}).get("value")
        or []
    )
    return [item for item in hits if isinstance(item, dict)]


def parse_search_results(raw_data: Any) -> list[dict[str, Any]]:
    """Convert raw API responses into a list of normalized result dictionaries."""
    results: list[dict[str, Any]] = []
    for item in _extract_hits(raw_data):
        date_text = str(item.get("date") or "")
        year_match = _YEAR_PATTERN.search(date_text)

        results.append({
            "title": item.get("title") or item.get("name") or "Untitled result",
            "url": item.get("url") or item.get("link") or "",
            "snippet": item.get("description") or item.get("snippet") or "",
            "source": item.get("domain") or item.get("source") or "",
            "date": item.get("date"),
            "year": item.get("year") or (int(year_match.group(0)) if year_match else None),
        })

    return results
