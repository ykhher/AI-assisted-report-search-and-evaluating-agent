"""General-purpose utilities."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def ensure_list(value: Any) -> List[Any]:
    """Ensure the return value is always a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def safe_get(mapping: Dict[Any, Any], key: Any, default: Any = None) -> Any:
    """Safe `dict.get` that won't fail on non-dict inputs."""
    if not isinstance(mapping, dict):
        return default
    return mapping.get(key, default)
