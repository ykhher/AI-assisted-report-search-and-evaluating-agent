"""Lightweight post-ranking verification helpers."""

from __future__ import annotations

import re
from typing import Any

from source.citation_extractor import assign_citations_to_claims, extract_citations
from source.claim_verifier import Claim, VerifiedClaim, VerificationResult
from source.scoring import compute_verification_adjusted_quality_score
from source.verification_metrics import compute_metrics_from_notes


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


def build_claim_objects(
    claims: list[str],
    source_url: str = "",
    claim_citations: dict[str, list[str]] | None = None,
) -> list[Claim]:
    """Convert extracted claim strings into deterministic verifier Claim objects."""
    claim_citations = claim_citations or {}
    claim_objects: list[Claim] = []
    for index, claim in enumerate(claims, start=1):
        numeric = bool(NUMERIC_PATTERN.search(claim))
        lowered = claim.lower()
        externally_checkable = numeric or any(term in lowered for term in FORECAST_TERMS)
        citations = list(claim_citations.get(claim, []))
        if not citations and source_url and externally_checkable:
            citations = [source_url]

        claim_objects.append(
            Claim(
                position=f"L1.S{index}",
                claim=claim,
                claim_type="A" if citations and externally_checkable else ("E" if not externally_checkable else "F"),
                rationale="Extracted by deterministic claim priority rules.",
                numeric=numeric,
                citations=citations,
                implicit_citations=[],
                cross_references=[],
            )
        )
    return claim_objects


def verify_claims_against_context(
    claims: list[str],
    text: str,
    url: str,
) -> list[VerifiedClaim]:
    """Verify extracted claims against the current report text without using an LLM."""
    claim_citations = assign_citations_to_claims(claims, text, report_url=url)
    claim_objects = build_claim_objects(claims, source_url=url, claim_citations=claim_citations)
    verified_claims: list[VerifiedClaim] = []
    local_support = any(term in str(text or "").lower() for term in REFERENCE_TERMS)

    for claim in claim_objects:
        if claim.claim_type == "E":
            result = "supported"
            explanation = "Claim is treated as common, structural, or opinion-like in this lightweight verifier."
        elif local_support:
            result = "supported"
            explanation = "The extracted report text contains local source or reference cues."
        else:
            result = "not_supported"
            explanation = "No obvious local source or reference cue was found in the extracted report text."

        verification_url = claim.citations[0] if claim.citations else url
        verified_claims.append(
            VerifiedClaim(
                claim=claim,
                verifications=[
                    VerificationResult(
                        claim=claim.claim,
                        explanation=explanation,
                        result=result,
                        url=verification_url,
                        reliable=True,
                        reliable_explanation="Verification used the report's own extracted text.",
                    )
                ] if claim.claim_type != "E" else [],
            )
        )

    return verified_claims


def attach_verification_notes(
    report: dict[str, Any],
    claims: list[str],
    text: str = "",
    signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach a small confidence note to each extracted claim."""
    enriched = dict(report)
    signals = dict(signals or {})
    source_text = str(text or signals.get("_text", ""))
    local_text = source_text.strip().lower()
    extracted_citations = extract_citations(source_text)
    claim_citations = assign_citations_to_claims(
        claims,
        source_text,
        report_url=str(enriched.get("url", "")),
    )
    verified_claims = verify_claims_against_context(
        claims,
        text=source_text,
        url=str(enriched.get("url", "")),
    )
    verified_by_claim = {item.claim.claim: item for item in verified_claims}

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

        has_measurable_claim = bool(NUMERIC_PATTERN.search(claim) or any(term in lowered for term in FORECAST_TERMS))
        verified_claim = verified_by_claim.get(claim)
        final_result = "supported" if has_local_support else "not_supported"
        final_explanation = ""
        if verified_claim is not None:
            final_result, final_explanation = verified_claim.final_result_and_explanation()

        if final_result == "supported" and has_local_support:
            claim_type = "A"
            result = "supported"
        elif has_measurable_claim:
            claim_type = "F"
            result = "unsupported"
        else:
            claim_type = "E"
            result = "supported"

        verifications = []
        if claim_type == "A":
            for citation_url in claim_citations.get(claim, []):
                verifications.append(
                    {
                        "url": citation_url,
                        "result": result,
                        "reliable": confidence in {"medium", "high"},
                    }
                )

        verification_notes.append(
            {
                "claim": claim,
                "claim_type": claim_type,
                "result": result,
                "numeric": bool(NUMERIC_PATTERN.search(claim)),
                "confidence": confidence,
                "notes": notes,
                "explanation": final_explanation,
                "verifications": verifications,
            }
        )

    summary = []
    if verification_notes:
        summary.append("lightweight verification pass completed")
    if not references_present:
        summary.append("verification is limited to internal text cues, not external source checking")

    enriched["verification_notes"] = verification_notes
    enriched["verified_claims"] = [item.to_dict() for item in verified_claims]
    enriched["extracted_citations"] = extracted_citations
    verification_metrics = compute_metrics_from_notes(
        verification_notes,
        report_url=str(enriched.get("url", "")),
    )
    enriched["verification_metrics"] = verification_metrics
    signals_with_text = dict(signals)
    signals_with_text.setdefault("_text", source_text)
    enriched["verification_adjusted_quality_score"] = compute_verification_adjusted_quality_score(
        signals_with_text,
        verification_metrics,
    )
    enriched["verification_summary"] = summary
    return enriched
