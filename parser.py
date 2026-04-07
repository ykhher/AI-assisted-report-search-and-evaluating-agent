"""Helpers for parsing raw search API responses into a normalized list."""

from __future__ import annotations

import re
from typing import Any, Dict, List


_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


def parse_search_results(raw_data: Any) -> List[Dict[str, Any]]:
    """Convert raw API responses into a list of normalized result dictionaries."""
    if isinstance(raw_data, list):
        hits = raw_data
    elif isinstance(raw_data, dict):
        hits = (
            raw_data.get("organic_results")
            or raw_data.get("hits")
            or raw_data.get("results")
            or raw_data.get("documents")
            or raw_data.get("data")
            or raw_data.get("webPages", {}).get("value")
            or []
        )
    else:
        hits = []

    results: List[Dict[str, Any]] = []
    for item in hits:
        if not isinstance(item, dict):
            continue

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