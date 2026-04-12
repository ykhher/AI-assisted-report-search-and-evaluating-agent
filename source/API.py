"""Helper for calling SearchAPI.com / searchapi.io for live web search."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


DEFAULT_API_URL = "https://www.searchapi.io/api/v1/search"
DEFAULT_API_KEY = "rLsVnoA7RPQpW6YLfHuU3FXi"


def search_market_reports(
    query: str,
    count: int = 10,
    language: str = "en",
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_API_URL,
) -> Dict[str, Any]:
    """Search for market-report style results using SearchAPI.com."""
    if api_key is None:
        api_key = os.environ.get("SEARCHAPI_API_KEY") or os.environ.get("YDC_API_KEY") or DEFAULT_API_KEY

    params = {
        "engine": "google",
        "q": query,
        "num": count,
        "hl": language,
        "api_key": api_key,
    }

    response = requests.get(base_url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    result = search_market_reports("Robot market")
    print(result)
