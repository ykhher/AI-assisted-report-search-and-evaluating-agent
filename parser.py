"""Helpers for parsing raw search results."""

from __future__ import annotations

from typing import Any, Dict, List
from urllib import response


def parse_search_results(raw_data: Any) -> List[Dict[str, Any]]:
    """Convert raw API responses into a list of normalized result dictionaries.

    This helps normalize across different search APIs (e.g., Elasticsearch-style hits,
    Bing WebPages results, or any other source that provides a list of result objects).
    """

    # If the response is already just a list of items, treat it as the hits list.
    if isinstance(raw_data, list):
        hits = raw_data
    elif isinstance(raw_data, dict):
        # Try common search response keys.
        hits = raw_data.get("hits")
        if not hits:
            hits = raw_data.get("results")
        if not hits:
            hits = raw_data.get("webPages", {}).get("value")
    else:
        hits = []

    hits = hits or []

    results: List[Dict[str, Any]] = []
    for item in hits:
        if not isinstance(item, dict):
            continue

        results.append({
            "title": item.get("title") or item.get("name"),
            "url": item.get("url"),
            "snippet": item.get("description") or item.get("snippet"),
            "source": item.get("domain"),
            "date": item.get("date"),
        })

    return results

data = response.json()
parsed = parse_search_results(data)

for r in parsed:
    print(r)