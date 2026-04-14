"""Small wrapper around the live search API used by the agent."""

from __future__ import annotations

import os
from typing import Any

import requests


DEFAULT_API_URL = "https://www.searchapi.io/api/v1/search"
DEFAULT_API_KEY = "rLsVnoA7RPQpW6YLfHuU3FXi"


def _resolve_api_key(api_key: str | None = None) -> str:
    """Resolve the API key from explicit input or environment variables."""
    key = (
        api_key
        or os.environ.get("SEARCHAPI_API_KEY")
        or os.environ.get("YDC_API_KEY")
        or DEFAULT_API_KEY
    )
    if key:
        return key
    raise ValueError("Missing search API key. Set SEARCHAPI_API_KEY or YDC_API_KEY.")


def search_market_reports(
    query: str,
    count: int = 20,
    language: str = "en",
    api_key: str | None = None,
    base_url: str = DEFAULT_API_URL,
) -> dict[str, Any]:
    """Call the live search API and return the raw JSON response."""
    params = {
        "engine": "google",
        "q": query,
        "num": count,
        "hl": language,
        "api_key": _resolve_api_key(api_key),
    }

    response = requests.get(base_url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()
