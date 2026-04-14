"""Small ranking helpers kept for backward compatibility."""

from __future__ import annotations

from typing import Any


def rank_results(
    results: list[dict[str, Any]],
    key: str = "score",
    reverse: bool = True,
) -> list[dict[str, Any]]:
    """Sort result dictionaries by one numeric-like field."""
    def sort_value(item: dict[str, Any]) -> float:
        try:
            return float(item.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    return sorted(results, key=sort_value, reverse=reverse)
