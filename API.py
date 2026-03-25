"""Minimal helper for calling the YDC index search endpoint."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


DEFAULT_API_URL = "https://ydc-index.io/v1/search"
DEFAULT_API_KEY = "ydc-sk-cae14558cea4101c-a20sHVoibuOI2ZLG6t366u6qaLJnEv3r-69818a82"


def search_market_reports(
    query: str,
    count: int = 10,
    language: str = "en",
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_API_URL,
) -> Dict[str, Any]:
    """Search for market report style results and return parsed JSON."""

    if api_key is None:
        api_key = os.environ.get("YDC_API_KEY", DEFAULT_API_KEY)

    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-KEY"] = api_key

    params = {
        "query": query,
        "count": count,
        "language": language,
    }

    response = requests.get(base_url, headers=headers, params=params, timeout=10)
    response.raise_for_status()

    return response.json()


if __name__ == "__main__":
    result = search_market_reports("AI market forecast report")
    print(result)
