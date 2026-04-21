"""Extract lightweight citations from report text.

This is deterministic and intentionally conservative. It focuses on URLs and
source/reference lines that can be turned into verification targets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from source.verification_metrics import clean_url


URL_PATTERN = re.compile(r"https?://[^\s\])}>\"']+|www\.[^\s\])}>\"']+", re.IGNORECASE)
BRACKET_CITATION_PATTERN = re.compile(r"\[(?P<id>\d{1,3})\]")
FOOTNOTE_CITATION_PATTERN = re.compile(r"(?<!\d)(?P<id>\d{1,3})(?!\d)")
REFERENCE_LINE_PATTERN = re.compile(
    r"^\s*(?:\[(?P<bracket>\d{1,3})\]|(?P<number>\d{1,3})[.)])\s+(?P<body>.+)$"
)
REFERENCE_HEADING_PATTERN = re.compile(r"^\s*(references|bibliography|sources|source notes)\s*:?\s*$", re.IGNORECASE)
SOURCE_LINE_PATTERN = re.compile(r"\b(source|sources|according to|adapted from|based on)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Citation:
    """One extracted citation target."""

    citation_id: str
    url: str
    label: str = ""
    source_text: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "citation_id": self.citation_id,
            "url": self.url,
            "label": self.label,
            "source_text": self.source_text,
        }


def _normalize_url(raw_url: str) -> str:
    value = str(raw_url or "").strip().rstrip(".,;:")
    if value.lower().startswith("www."):
        value = "https://" + value
    return clean_url(value)


def _line_id(index: int) -> str:
    return f"url:{index}"


def extract_citations(text: str) -> dict[str, str]:
    """Return citation id -> URL extracted from report text."""
    citations: dict[str, str] = {}
    lines = str(text or "").splitlines()
    in_references = False
    url_index = 1

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if REFERENCE_HEADING_PATTERN.match(stripped):
            in_references = True
            continue

        urls = [_normalize_url(match.group(0)) for match in URL_PATTERN.finditer(stripped)]
        urls = [url for url in urls if url]
        if not urls:
            if in_references and len(stripped.split()) <= 3:
                in_references = False
            continue

        reference_match = REFERENCE_LINE_PATTERN.match(stripped)
        if reference_match:
            citation_id = reference_match.group("bracket") or reference_match.group("number")
            if citation_id:
                citations[citation_id] = urls[0]

        if in_references or SOURCE_LINE_PATTERN.search(stripped):
            for url in urls:
                citations.setdefault(_line_id(url_index), url)
                url_index += 1

        for url in urls:
            citations.setdefault(url, url)

    return citations


def extract_citation_records(text: str) -> list[Citation]:
    """Return detailed citation records useful for debugging/export."""
    citation_map = extract_citations(text)
    return [
        Citation(citation_id=citation_id, url=url, label=citation_id, source_text="")
        for citation_id, url in citation_map.items()
    ]


def _sentence_citation_ids(sentence: str) -> list[str]:
    ids = [match.group("id") for match in BRACKET_CITATION_PATTERN.finditer(sentence)]
    if ids:
        return ids

    # Only treat bare footnote-like numbers as citation IDs when they appear at
    # the end of a sentence or near punctuation. This avoids grabbing data values.
    tail = sentence[-12:]
    return [
        match.group("id")
        for match in FOOTNOTE_CITATION_PATTERN.finditer(tail)
        if int(match.group("id")) <= 100
    ]


def assign_citations_to_claims(
    claims: list[str],
    text: str,
    report_url: str = "",
) -> dict[str, list[str]]:
    """Map each claim string to citation URLs found near or inside that claim."""
    citation_map = extract_citations(text)
    claim_to_urls: dict[str, list[str]] = {}
    default_urls = list(dict.fromkeys(citation_map.values()))
    if not default_urls and report_url:
        default_urls = [clean_url(report_url)]

    for claim in claims:
        urls: list[str] = []
        for citation_id in _sentence_citation_ids(claim):
            url = citation_map.get(citation_id)
            if url:
                urls.append(url)

        for raw_url in URL_PATTERN.findall(claim):
            url = _normalize_url(raw_url)
            if url:
                urls.append(url)

        # If the claim text does not carry an explicit marker, use extracted
        # source URLs as implicit citations. Limit to a few targets to keep
        # deterministic verification cheap and explainable.
        if not urls:
            urls.extend(default_urls[:3])

        claim_to_urls[claim] = list(dict.fromkeys(urls))

    return claim_to_urls


def citation_debug_payload(text: str, claims: list[str], report_url: str = "") -> dict[str, Any]:
    """Return a compact payload for inspecting citation extraction."""
    return {
        "citations": extract_citations(text),
        "claim_citations": assign_citations_to_claims(claims, text, report_url=report_url),
    }
