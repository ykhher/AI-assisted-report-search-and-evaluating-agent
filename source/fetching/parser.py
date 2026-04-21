"""Helpers for parsing raw search API responses into a normalized list."""

from __future__ import annotations

import re
from typing import Any


_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


def _reconstruct_openalex_abstract(index: Any) -> str:
    """Rebuild OpenAlex abstract text from an inverted index."""
    if not isinstance(index, dict):
        return ""

    positioned_words: list[tuple[int, str]] = []
    for word, positions in index.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            try:
                positioned_words.append((int(position), str(word)))
            except (TypeError, ValueError):
                continue

    positioned_words.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned_words)


def _openalex_location(item: dict[str, Any]) -> dict[str, Any]:
    """Return the best available OpenAlex location dictionary."""
    primary = item.get("primary_location")
    if isinstance(primary, dict):
        return primary

    locations = item.get("locations")
    if isinstance(locations, list):
        for location in locations:
            if isinstance(location, dict):
                return location
    return {}


def _openalex_url(item: dict[str, Any]) -> str:
    """Choose a stable URL for an OpenAlex work."""
    location = _openalex_location(item)
    pdf_url = location.get("pdf_url")
    landing_page_url = location.get("landing_page_url")
    open_access = item.get("open_access")
    if isinstance(open_access, dict):
        pdf_url = pdf_url or open_access.get("oa_url")

    for value in (pdf_url, landing_page_url, item.get("doi"), item.get("id")):
        cleaned = str(value or "").strip()
        if cleaned:
            return cleaned
    return ""


def _openalex_source(item: dict[str, Any]) -> str:
    """Extract source or host venue from an OpenAlex work."""
    location = _openalex_location(item)
    source = location.get("source")
    if isinstance(source, dict):
        name = source.get("display_name") or source.get("host_organization_name")
        if name:
            return str(name)

    host_venue = item.get("host_venue")
    if isinstance(host_venue, dict):
        name = host_venue.get("display_name") or host_venue.get("publisher")
        if name:
            return str(name)
    return ""


def _is_openalex_work(item: dict[str, Any]) -> bool:
    """Return True when a result row looks like an OpenAlex work."""
    return (
        "abstract_inverted_index" in item
        or "primary_location" in item
        or str(item.get("id", "")).startswith("https://openalex.org/")
    )


def _parse_openalex_work(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize one OpenAlex work to the internal search-result schema."""
    abstract = _reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
    title = item.get("title") or item.get("display_name") or "Untitled result"
    year = item.get("publication_year")
    if year is None:
        date_text = str(item.get("publication_date") or "")
        year_match = _YEAR_PATTERN.search(date_text)
        year = int(year_match.group(0)) if year_match else None

    work_type = item.get("type") or item.get("type_crossref") or ""
    snippet_parts = [abstract]
    if work_type:
        snippet_parts.append(f"OpenAlex type: {work_type}.")

    return {
        "title": title,
        "url": _openalex_url(item),
        "snippet": " ".join(part for part in snippet_parts if part).strip(),
        "source": _openalex_source(item),
        "date": item.get("publication_date"),
        "year": year,
        "is_pdf": ".pdf" in _openalex_url(item).lower(),
    }


def _extract_hits(raw_data: Any) -> list[dict[str, Any]]:
    """Extract a list of result dictionaries from common API response shapes."""
    if isinstance(raw_data, list):
        return [item for item in raw_data if isinstance(item, dict)]
    if not isinstance(raw_data, dict):
        return []

    hits = (
        raw_data.get("organic_results")
        or raw_data.get("hits")
        or raw_data.get("results")
        or raw_data.get("documents")
        or raw_data.get("data")
        or raw_data.get("webPages", {}).get("value")
        or []
    )
    return [item for item in hits if isinstance(item, dict)]


def parse_search_results(raw_data: Any) -> list[dict[str, Any]]:
    """Convert raw API responses into a list of normalized result dictionaries."""
    results: list[dict[str, Any]] = []
    for item in _extract_hits(raw_data):
        if _is_openalex_work(item):
            results.append(_parse_openalex_work(item))
            continue

        date_text = str(item.get("date") or "")
        year_match = _YEAR_PATTERN.search(date_text)

        results.append({
            "title": item.get("title") or item.get("name") or "Untitled result",
            "url": item.get("url") or item.get("link") or "",
            "snippet": item.get("description") or item.get("snippet") or "",
            "source": item.get("domain") or item.get("source") or "",
            "date": item.get("date"),
            "year": item.get("year") or (int(year_match.group(0)) if year_match else None),
        })

    return results
