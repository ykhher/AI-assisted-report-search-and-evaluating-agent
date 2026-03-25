"""Ranking utilities for search results."""

from __future__ import annotations

from typing import Dict, List


def rank_results(results: List[Dict[str, Any]], key: str = "score", reverse: bool = True) -> List[Dict[str, Any]]:
    """Sort results by a given key (default: score)."""
    return sorted(results, key=lambda r: r.get(key, 0), reverse=reverse)
