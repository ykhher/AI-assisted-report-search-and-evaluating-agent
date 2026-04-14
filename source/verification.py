"""Lightweight post-ranking verification helpers."""

from __future__ import annotations

import re
from typing import Any


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
NUMERIC_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
FORECAST_TERMS = ("forecast", "projected", "expected", "will reach", "will grow", "growth", "cagr", "adoption", "increase", "decrease")
METHODOLOGY_TERMS = ("methodology", "sample", "survey", "modeled", "modelled", "assumption", "scenario", "estimated")
REFERENCE_TERMS = ("reference", "references", "source", "sources", "bibliography", "dataset", "according to", "[", "http")


def _split_sentences(text: str) -> list[str]:
    sentences = [" ".join(sentence.split()) for sentence in SENTENCE_SPLIT_PATTERN.split(str(text or "").strip())]
    return [sentence for sentence in sentences if len(sentence.split()) >= 6]


def _claim_priority(sentence: str) -> tuple[int, int, int]:
    lowered = sentence.lower()
    has_numeric = int(bool(NUMERIC_PATTERN.search(sentence)))
    has_forecast = int(any(term in lowered for term in FORECAST_TERMS))
    # Shorter sentences are usually easier to read in the CLI.
    return (has_numeric, has_forecast, -abs(len(sentence.split()) - 18))


def extract_key_claims(text: str, max_claims: int = 3) -> list[str]:
    """Pick a few likely claims from the text."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    ranked = sorted(sentences, key=_claim_priority, reverse=True)
    claims: list[str] = []
    seen: set[str] = set()

    for sentence in ranked:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        if not NUMERIC_PATTERN.search(sentence) and not any(term in normalized for term in FORECAST_TERMS):
            continue
        claims.append(sentence)
        seen.add(normalized)
        if len(claims) >= max(1, max_claims):
            break

    return claims or ranked[: max(1, max_claims)]


def attach_verification_notes(
    report: dict[str, Any],
    claims: list[str],
    text: str = "",
    signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach a small confidence note to each extracted claim."""
    enriched = dict(report)
    signals = dict(signals or {})
    local_text = str(text or signals.get("_text", "")).strip().lower()

    methodology_present = bool(signals.get("methodology")) or bool(signals.get("parsed_has_methodology"))
    references_present = bool(signals.get("citation_score")) or bool(signals.get("parsed_has_references"))

    verification_notes: list[dict[str, Any]] = []
    for claim in claims:
        lowered = claim.lower()
        notes: list[str] = []
        confidence = "low"

        if NUMERIC_PATTERN.search(claim) or any(term in lowered for term in FORECAST_TERMS):
            notes.append("contains measurable forecast language")
            confidence = "medium"

        if methodology_present or any(term in lowered for term in METHODOLOGY_TERMS):
            notes.append("claim appears methodology-dependent")
            confidence = "high" if confidence == "medium" else "medium"

        has_local_support = references_present or any(term in local_text for term in REFERENCE_TERMS)
        if has_local_support:
            notes.append("extracted text shows some local reference or source support")
        else:
            notes.append("claim lacks obvious supporting reference in extracted text")
            if confidence == "high":
                confidence = "medium"

        verification_notes.append({"claim": claim, "confidence": confidence, "notes": notes})

    summary = []
    if verification_notes:
        summary.append("lightweight verification pass completed")
    if not references_present:
        summary.append("verification is limited to internal text cues, not external source checking")

    enriched["verification_notes"] = verification_notes
    enriched["verification_summary"] = summary
    return enriched
