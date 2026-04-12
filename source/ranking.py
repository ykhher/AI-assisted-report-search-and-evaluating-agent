"""Ranking utilities for search results."""

from __future__ import annotations

from typing import Any


def rank_results(results: list[dict[str, Any]], key: str = "score", reverse: bool = True) -> list[dict[str, Any]]:
    """Sort results by a given key (default: score)."""
    return sorted(results, key=lambda r: r.get(key, 0), reverse=reverse)
