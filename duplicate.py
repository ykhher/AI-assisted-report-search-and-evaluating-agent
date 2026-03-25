"""Helpers for deduplicating search results."""

from __future__ import annotations

from typing import Any, Dict, List


def deduplicate(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a new list of results with duplicates (by URL) removed."""

    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []

    for r in results:
        url = r.get("url")
        if not url or url in seen:
            continue

        unique.append(r)
        seen.add(url)

    return unique
