"""Small wrapper around the live SerpApi search API used by the agent."""

from __future__ import annotations

import os
from typing import Any

import requests


DEFAULT_API_URL = "https://serpapi.com/search.json"


def _resolve_api_key(api_key: str | None = None) -> str:
    """Resolve the SerpApi key from explicit input or environment variables."""
    key = api_key or os.environ.get("SERPAPI_API_KEY")
    if key:
        return key
    raise ValueError("Missing SerpApi key. Set SERPAPI_API_KEY.")


def search_market_reports(
    query: str,
    count: int = 20,
    language: str = "en",
    api_key: str | None = None,
    base_url: str = DEFAULT_API_URL,
) -> dict[str, Any]:
    """Call SerpApi Google Search and return the raw JSON response."""
    params: dict[str, Any] = {
        "engine": "google",
        "q": query,
        "num": max(1, min(int(count), 100)),
        "hl": language,
        "api_key": _resolve_api_key(api_key),
    }

    response = requests.get(base_url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()
