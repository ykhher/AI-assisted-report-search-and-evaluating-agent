"""Phase 4 scoring and ranking for report credibility.

This module exposes explicit, interpretable sub-scores:
- relevance_score
- report_validity_score
- quality_score
- authority_score
- final_score

It also keeps backward-compatible helpers used by the existing pipeline.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Mapping

# Allow direct execution with `python source/scoring.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.extractor import bottom_reference_score, footnote_score, institution_score
from local_qwen import assess_text_signals

_SECTION_KEYWORDS = ("introduction", "methodology", "results", "conclusion")
_CLAIM_KEYWORDS = ("increase", "decrease", "forecast", "projected", "expected")
_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")

_GENERIC_QUERY_TERMS = {
    "pdf", "report", "reports", "industry", "analysis", "market", "forecast",
    "outlook", "cagr", "projection", "research", "study", "overview",
}

_REPORT_HINT_TERMS = (
    "report", "whitepaper", "benchmark", "survey", "analysis", "outlook",
    "index", "study", "research", "findings", "annual", "global",
)

_REPORT_LIKE_TYPES = {
    "report",
    "research_report",
    "benchmark",
    "survey",
    "whitepaper",
    "research_note",
    "working_paper",
    "deck",
}

_SOURCE_CLASS_MAP = {
    "high": 0.9,
    "trusted": 0.9,
    "medium": 0.6,
    "standard": 0.6,
    "low": 0.35,
    "unknown": 0.45,
}


# Final ranking weights over clear sub-scores.
# Balanced for live web retrieval: snippets often lack methodology/citation text,
# so relevance, report-validity, and authority need meaningful influence.
FINAL_WEIGHTS: dict[str, float] = {
    "relevance_score": 0.35,
    "report_validity_score": 0.20,
    "quality_score": 0.30,
    "authority_score": 0.15,
}


def _clamp01(value: Any) -> float:
    """Convert a value to float and clamp it to [0, 1]."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(numeric, 1.0))


def _word_count(text: str) -> int:
    """Count words in a text string."""
    return len(re.findall(r"\b\w+\b", str(text).lower()))


def _extract_parsed(signals: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build a normalized parsed dict from signal fields if available."""
    signals = signals or {}
    section_lengths = signals.get("parsed_section_lengths", {})
    if not isinstance(section_lengths, dict):
        section_lengths = {}

    return {
        "section_lengths": section_lengths,
        "word_count": int(signals.get("parsed_word_count", 0) or 0),
        "has_methodology": bool(signals.get("parsed_has_methodology", False)),
        "has_references": bool(signals.get("parsed_has_references", False)),
        "has_statistics_language": bool(signals.get("parsed_has_statistics_language", False)),
    }


def _contains_report_hints(text: str) -> bool:
    """Return True when text looks like report-style content."""
    lowered = str(text).lower()
    return any(term in lowered for term in _REPORT_HINT_TERMS)


def _is_report_like_type(value: Any) -> bool:
    """Return True for classifier labels that represent report-like documents."""
    return str(value or "").strip().lower() in _REPORT_LIKE_TYPES


def _nested_metric(payload: Mapping[str, Any] | None, path: tuple[str, ...], default: float = 0.0) -> float:
    """Read a nested numeric verification metric safely."""
    current: Any = payload or {}
    for key in path:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
    return _clamp01(current)


def _scaled_metric(value: Any, scale: float) -> float:
    """Scale an unbounded positive metric into [0, 1]."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if scale <= 0:
        return _clamp01(numeric)
    return _clamp01(numeric / scale)


def _verification_quality_components(metrics: Mapping[str, Any] | None) -> dict[str, float] | None:
    """Map verification metrics onto the six quality sub-components.

    The verifier produces integrity/sufficiency metrics after claims are
    checked. This function folds those metrics back into the same conceptual
    buckets used by `quality_score`:
    methodology, citation/reference, consistency, structure, data, and claim.
    """
    if not isinstance(metrics, Mapping):
        return None

    raw_claims = metrics.get("raw", {}).get("claim_metrics", {}) if isinstance(metrics.get("raw"), Mapping) else {}
    total_claims = 0
    if isinstance(raw_claims, Mapping):
        try:
            total_claims = int(raw_claims.get("total_claims", 0) or 0)
        except (TypeError, ValueError):
            total_claims = 0
    if total_claims <= 0:
        return None

    claim_accuracy = _nested_metric(metrics, ("integrity", "claim_factuality", "claim_accuracy"))
    external_claim_accuracy = _nested_metric(metrics, ("integrity", "claim_factuality", "external_claim_accuracy"))
    external_numeric_claim_accuracy = _nested_metric(
        metrics,
        ("integrity", "claim_factuality", "external_numeric_claim_accuracy"),
    )
    citation_accuracy = _nested_metric(metrics, ("integrity", "citation_validity", "citation_accuracy"))
    supported_per_shown = _nested_metric(metrics, ("integrity", "reference_accuracy", "supported_per_shown"))
    supported_per_used = _nested_metric(metrics, ("integrity", "reference_accuracy", "supported_per_used"))
    used_per_shown = _nested_metric(metrics, ("integrity", "reference_accuracy", "used_per_shown"))
    reproducibility = _nested_metric(metrics, ("integrity", "reference_quality", "reproducibility"))
    reliability = _nested_metric(metrics, ("integrity", "reference_quality", "reliability"))
    diversity_hhi = _scaled_metric(
        metrics.get("integrity", {}).get("reference_diversity", {}).get("diversity_hhi", 0.0)
        if isinstance(metrics.get("integrity"), Mapping)
        else 0.0,
        10.0,
    )
    verified_claims_ratio = _nested_metric(metrics, ("sufficiency", "source_support", "verified_claims_ratio"))
    average_citations_per_claim = _scaled_metric(
        metrics.get("sufficiency", {}).get("source_support", {}).get("average_citations_per_claim", 0.0)
        if isinstance(metrics.get("sufficiency"), Mapping)
        else 0.0,
        2.0,
    )

    return {
        "methodology": round((reproducibility + reliability + verified_claims_ratio) / 3.0, 3),
        "citation": round(
            (
                0.30 * citation_accuracy
                + 0.20 * supported_per_shown
                + 0.20 * supported_per_used
                + 0.10 * used_per_shown
                + 0.10 * reliability
                + 0.10 * diversity_hhi
            ),
            3,
        ),
        "consistency": round((claim_accuracy + external_claim_accuracy) / 2.0, 3),
        "structure": round((used_per_shown + verified_claims_ratio) / 2.0, 3),
        "data_support": round(external_numeric_claim_accuracy, 3),
        "claim_density": round(
            (
                0.40 * verified_claims_ratio
                + 0.30 * average_citations_per_claim
                + 0.30 * external_claim_accuracy
            ),
            3,
        ),
    }


def _blend_quality_component(base: float, verification: float, verification_weight: float) -> float:
    """Blend a base quality component with its verification-derived counterpart."""
    return _clamp01((1.0 - verification_weight) * _clamp01(base) + verification_weight * _clamp01(verification))


def compute_structure_score(text: str) -> float:
    """Measure whether the report contains standard structural sections."""
    cleaned = str(text).lower()
    detected_sections = sum(1 for section in _SECTION_KEYWORDS if section in cleaned)
    return round(detected_sections / len(_SECTION_KEYWORDS), 3)


def compute_claim_density(text: str) -> float:
    """Measure how densely the text uses analytical claim language."""
    cleaned = str(text).lower()
    total_words = _word_count(cleaned)
    if total_words == 0:
        return 0.0

    keyword_count = sum(
        len(re.findall(fr"\b{re.escape(keyword)}\b", cleaned))
        for keyword in _CLAIM_KEYWORDS
    )
    return round(min((keyword_count / total_words) / 0.01, 1.0), 3)


def in_text_citation_score(text: str) -> float:
    """Backward-compatible wrapper for inline footnote detection."""
    return footnote_score(text)


def reference_section_score(text: str) -> float:
    """Backward-compatible wrapper for numbered bottom-reference detection."""
    return bottom_reference_score(text)


def numbered_reference_score(text: str) -> float:
    """Backward-compatible wrapper for numbered bottom-reference detection."""
    return bottom_reference_score(text)


def compute_citation_score(text: str) -> float:
    """Blend heuristic and local-LLM evidence for references and source support."""
    heuristic_score = max(bottom_reference_score(text), footnote_score(text))
    heuristic_score += 0.2 * institution_score(text)

    llm_scores = assess_text_signals(str(text))
    reference_llm = _clamp01(llm_scores.get("reference_score", 0.0))
    return round(min(max(heuristic_score, reference_llm), 1.0), 3)


def compute_consistency_score(text: str) -> float:
    """Blend structural consistency heuristics with local LLM judgement."""
    lowered = str(text).lower()
    detected_sections = [section for section in _SECTION_KEYWORDS if section in lowered]

    heuristic_score = 0.0
    if detected_sections:
        valid_sections = 0
        for section in detected_sections:
            start = lowered.find(section)
            next_positions = [
                lowered.find(other, start + 1)
                for other in _SECTION_KEYWORDS
                if lowered.find(other, start + 1) != -1
            ]
            end = min(next_positions) if next_positions else len(lowered)
            segment = lowered[start:end]

            has_number = bool(_NUMBER_PATTERN.search(segment))
            has_claim = any(keyword in segment for keyword in _CLAIM_KEYWORDS)
            if has_number or has_claim:
                valid_sections += 1

        heuristic_score = round(valid_sections / len(detected_sections), 3)

    llm_scores = assess_text_signals(str(text))
    consistency_llm = _clamp01(llm_scores.get("consistency_score", 0.0))
    return round(max(heuristic_score, consistency_llm), 3)


def compute_relevance_score(query: str, text: str) -> float:
    """Compute keyword-overlap relevance normalized to [0,1].

    Formula:
    overlap / number_of_non_generic_query_terms
    """
    query_words = {
        word for word in re.findall(r"[a-z0-9]+", str(query).lower())
        if len(word) > 2 and word not in _GENERIC_QUERY_TERMS
    }
    if not query_words:
        query_words = {word for word in re.findall(r"[a-z0-9]+", str(query).lower()) if len(word) > 2}
    if not query_words:
        return 0.0

    text_words = set(re.findall(r"[a-z0-9]+", str(text).lower()))
    overlap = len(query_words & text_words)
    return round(min(overlap / len(query_words), 1.0), 3)


def compute_report_validity_score(doc: Mapping[str, Any], parsed: Mapping[str, Any] | None = None, classifier_validity: float | None = None) -> float:
    """Estimate whether an item is a valid report rather than weak snippet content.

    Uses full-text parse metadata when available, and falls back to light text checks.
    
    If classifier_validity is provided (from report_classifier), blends it with heuristic score.

    Formula:
    0.35 * length_component
    + 0.25 * report_format_component
    + 0.20 * section_component
    + 0.20 * evidence_component
    
    If classifier_validity available: blend both scores with 0.4 weight on classifier score.
    """
    parsed = dict(parsed or {})
    text = str(doc.get("text") or doc.get("snippet") or "")
    title = str(doc.get("title") or "")
    url = str(doc.get("url") or "")

    parsed_word_count = int(parsed.get("word_count", 0) or 0)
    word_count = parsed_word_count if parsed_word_count > 0 else _word_count(text)
    has_full_text = word_count >= 250
    length_component = min(word_count / 800.0, 1.0) if has_full_text else min(word_count / 160.0, 0.45)

    is_pdf = bool(doc.get("is_pdf", False) or url.lower().endswith(".pdf"))
    has_report_hints = _contains_report_hints(f"{title} {text} {url}")
    report_format_component = 1.0 if (is_pdf or has_report_hints) else 0.35

    section_lengths = parsed.get("section_lengths", {})
    if isinstance(section_lengths, dict) and section_lengths:
        present = sum(1 for key in _SECTION_KEYWORDS if int(section_lengths.get(key, 0) or 0) > 0)
        section_component = present / len(_SECTION_KEYWORDS)
    else:
        section_component = compute_structure_score(text)

    evidence_flags = [
        1.0 if parsed.get("has_methodology", False) else 0.0,
        1.0 if parsed.get("has_references", False) else 0.0,
        1.0 if parsed.get("has_statistics_language", False) else 0.0,
    ]
    if any(evidence_flags):
        evidence_component = sum(evidence_flags) / len(evidence_flags)
    else:
        inferred = [
            1.0 if "method" in text.lower() else 0.0,
            1.0 if compute_citation_score(text) >= 0.3 else 0.0,
            1.0 if bool(_NUMBER_PATTERN.search(text)) else 0.0,
        ]
        evidence_component = sum(inferred) / len(inferred)

    score = (
        0.35 * _clamp01(length_component)
        + 0.25 * _clamp01(report_format_component)
        + 0.20 * _clamp01(section_component)
        + 0.20 * _clamp01(evidence_component)
    )

    # SERP snippets and PDF landing pages often do not expose full sections.
    # Give clearly report-like short results a modest floor, but still keep
    # true low-information pages below fully parsed reports.
    if not has_full_text and has_report_hints:
        snippet_floor = 0.42 if is_pdf else 0.34
        score = max(score, snippet_floor)
    
    # NEW: Blend with classifier validity if provided
    if classifier_validity is not None:
        classifier_val = _clamp01(classifier_validity)
        score = 0.6 * score + 0.4 * classifier_val
    
    return round(_clamp01(score), 3)


def compute_quality_score(signals: Mapping[str, Any], parsed: Mapping[str, Any] | None = None) -> float:
    """Compute intrinsic analytical quality from extracted signals.

    Formula:
    0.22 * methodology
    + 0.22 * citation
    + 0.18 * consistency
    + 0.14 * structure
    + 0.14 * data_support
    + 0.10 * claim_density
    """
    parsed = dict(parsed or {})
    raw_text = str(signals.get("_text", ""))
    source_context = str(signals.get("source_name", signals.get("source_label", "")))
    llm_scores = assess_text_signals(raw_text, source=source_context)

    methodology = max(
        _clamp01(signals.get("methodology", signals.get("has_methodology", 0))),
        _clamp01(llm_scores.get("methodology_score", 0.0)),
        1.0 if parsed.get("has_methodology", False) else 0.0,
    )
    citation = max(
        _clamp01(signals.get("citation_score", compute_citation_score(raw_text))),
        1.0 if parsed.get("has_references", False) else 0.0,
    )
    consistency = max(
        _clamp01(signals.get("consistency_score", compute_consistency_score(raw_text))),
        0.75 if parsed.get("has_statistics_language", False) else 0.0,
    )
    structure = _clamp01(signals.get("structure_score", compute_structure_score(raw_text)))
    data_density = _clamp01(signals.get("data_density", 0.0))
    data_support = round(data_density ** 1.3, 3)
    claim_density = _clamp01(signals.get("claim_density", compute_claim_density(raw_text)))

    verification_components = _verification_quality_components(signals.get("verification_metrics"))
    if verification_components:
        methodology = _blend_quality_component(methodology, verification_components["methodology"], 0.20)
        citation = _blend_quality_component(citation, verification_components["citation"], 0.35)
        consistency = _blend_quality_component(consistency, verification_components["consistency"], 0.30)
        structure = _blend_quality_component(structure, verification_components["structure"], 0.15)
        data_support = _blend_quality_component(data_support, verification_components["data_support"], 0.30)
        claim_density = _blend_quality_component(claim_density, verification_components["claim_density"], 0.30)

    score = (
        0.22 * methodology
        + 0.22 * citation
        + 0.18 * consistency
        + 0.14 * structure
        + 0.14 * data_support
        + 0.10 * claim_density
    )

    report_type = signals.get("report_type")
    classifier_validity = _clamp01(signals.get("report_validity_score_classifier", 0.0))
    report_like = _is_report_like_type(report_type) or _contains_report_hints(raw_text)
    word_count = _word_count(raw_text)

    # When only SERP snippets or blocked PDF landing pages are available, the
    # hard evidence signals above are often absent even for useful reports.
    # Use a capped proxy floor so report-like, recent, authoritative snippets
    # are not crushed to zero, while weak pages still need actual evidence.
    if report_like and word_count < 500:
        authority_proxy = max(
            _clamp01(signals.get("authority_prior", 0.0)),
            _clamp01(signals.get("source", 0.0)),
            _clamp01(signals.get("llm_source_score", 0.0)),
        )
        recency_proxy = _clamp01(signals.get("recency", 0.0))
        snippet_proxy = (
            0.18
            + 0.12 * classifier_validity
            + 0.10 * authority_proxy
            + 0.08 * recency_proxy
            + 0.07 * data_density
            + 0.05 * claim_density
        )
        score = max(score, min(snippet_proxy, 0.52))

    return round(_clamp01(score), 3)


def compute_verification_adjusted_quality_score(
    signals: Mapping[str, Any],
    verification_metrics: Mapping[str, Any],
    parsed: Mapping[str, Any] | None = None,
) -> float:
    """Recompute quality after verification metrics are available."""
    enriched_signals = dict(signals or {})
    enriched_signals["verification_metrics"] = dict(verification_metrics or {})
    return compute_quality_score(enriched_signals, parsed=parsed)


def compute_verification_adjusted_final_score(report: Mapping[str, Any]) -> dict[str, float]:
    """Return score_breakdown using verification-adjusted quality when available."""
    breakdown = dict(report.get("score_breakdown", {}) or {})
    adjusted_quality = report.get("verification_adjusted_quality_score")
    if adjusted_quality is None:
        return {
            "relevance_score": _clamp01(breakdown.get("relevance_score", report.get("relevance_score", 0.0))),
            "report_validity_score": _clamp01(breakdown.get("report_validity_score", report.get("report_validity_score", 0.0))),
            "quality_score": _clamp01(breakdown.get("quality_score", report.get("quality_score", 0.0))),
            "authority_score": _clamp01(breakdown.get("authority_score", report.get("authority_score", 0.0))),
            "final_score": _clamp01(breakdown.get("final_score", report.get("score", 0.0))),
        }

    adjusted_breakdown = {
        "relevance_score": _clamp01(breakdown.get("relevance_score", report.get("relevance_score", 0.0))),
        "report_validity_score": _clamp01(breakdown.get("report_validity_score", report.get("report_validity_score", 0.0))),
        "quality_score": _clamp01(adjusted_quality),
        "authority_score": _clamp01(breakdown.get("authority_score", report.get("authority_score", 0.0))),
    }
    adjusted_breakdown["final_score"] = compute_final_score(adjusted_breakdown)
    return adjusted_breakdown


def compute_authority_score(source: Any, source_class: str | None = None, authority_prior: float | None = None) -> float:
    """Estimate source authority from multiple signals with priority order.

    Priority:
    1) explicit authority_prior from source_classifier (most recent/accurate)
    2) explicit source_class mapping (high/medium/low)
    3) numeric source score if already provided
    4) simple domain heuristic from source string
    """
    # NEW: Use authority_prior from source_classifier if available
    if authority_prior is not None:
        prior = _clamp01(authority_prior)
        if prior > 0:
            return round(prior, 3)
    
    if source_class:
        mapped = _SOURCE_CLASS_MAP.get(str(source_class).strip().lower())
        if mapped is not None:
            return round(_clamp01(mapped), 3)

    numeric_source = _clamp01(source)
    if numeric_source > 0:
        return round(numeric_source, 3)

    source_text = str(source or "").lower()
    if not source_text:
        return 0.35

    high_markers = (".gov", ".edu", "worldbank", "oecd", "imf", "mckinsey", "bcg")
    medium_markers = (".org", "research", "institute", "consulting")

    if any(marker in source_text for marker in high_markers):
        return 0.9
    if any(marker in source_text for marker in medium_markers):
        return 0.65
    return 0.45


def compute_final_score(score_dict: Mapping[str, Any]) -> float:
    """Combine interpretable sub-scores into one final rank score.

    Formula:
    final = w_r * relevance + w_v * validity + w_q * quality + w_a * authority

    A mild penalty is applied when report_validity_score is very low.
    """
    relevance_score = _clamp01(score_dict.get("relevance_score", 0.0))
    validity_score = _clamp01(score_dict.get("report_validity_score", 0.0))
    quality_score = _clamp01(score_dict.get("quality_score", 0.0))
    authority_score = _clamp01(score_dict.get("authority_score", 0.0))

    weighted = (
        FINAL_WEIGHTS["relevance_score"] * relevance_score
        + FINAL_WEIGHTS["report_validity_score"] * validity_score
        + FINAL_WEIGHTS["quality_score"] * quality_score
        + FINAL_WEIGHTS["authority_score"] * authority_score
    )

    return round(_clamp01(weighted), 3)


def compute_report_scores(report: Mapping[str, Any], query: str | None = None) -> dict[str, float]:
    """Compute the full structured score dictionary for one report.

    Returns keys:
    - relevance_score
    - report_validity_score
    - quality_score
    - authority_score
    - final_score
    
    Uses classification outputs from signals (source_class, authority_prior, report_validity_score_classifier)
    if available.
    """
    signals = dict(report.get("signals", {}) or {})
    parsed = _extract_parsed(signals)

    text = str(report.get("text") or report.get("snippet") or signals.get("_text", ""))
    local_doc = {
        "title": report.get("title", ""),
        "url": report.get("url", ""),
        "text": text,
        "snippet": report.get("snippet", ""),
        "is_pdf": bool(report.get("is_pdf", signals.get("is_pdf", False))),
    }

    if report.get("relevance") is not None:
        relevance_score = _clamp01(report.get("relevance", 0.0))
    elif query:
        relevance_score = compute_relevance_score(query, text)
    else:
        relevance_score = 0.0

    # NEW: Pass classifier_validity to report validity computation
    classifier_validity = signals.get("report_validity_score_classifier")
    validity_score = compute_report_validity_score(local_doc, parsed, classifier_validity=classifier_validity)
    
    quality_score = compute_quality_score(signals, parsed)

    source_value = signals.get("source", report.get("source", signals.get("source_name", "")))
    # NEW: Pass authority_prior from classifier
    authority_prior = signals.get("authority_prior")
    authority_score = compute_authority_score(
        source_value, 
        source_class=report.get("source_class", signals.get("source_class")),
        authority_prior=authority_prior
    )

    score_dict = {
        "relevance_score": round(_clamp01(relevance_score), 3),
        "report_validity_score": round(_clamp01(validity_score), 3),
        "quality_score": round(_clamp01(quality_score), 3),
        "authority_score": round(_clamp01(authority_score), 3),
    }
    score_dict["final_score"] = compute_final_score(score_dict)
    return score_dict


def compute_rqi(signals: dict, text: str = "") -> float:
    """Backward-compatible RQI wrapper.

    RQI now represents report quality emphasis with report-validity support:
    RQI = 0.70 * quality_score + 0.30 * report_validity_score
    """
    merged_signals = dict(signals or {})
    if text and not merged_signals.get("_text"):
        merged_signals["_text"] = text

    parsed = _extract_parsed(merged_signals)
    quality = compute_quality_score(merged_signals, parsed)
    validity = compute_report_validity_score({"text": merged_signals.get("_text", text)}, parsed)

    rqi = round(_clamp01(0.70 * quality + 0.30 * validity), 3)

    signals["quality_score"] = quality
    signals["report_validity_score"] = validity
    signals["RQI"] = rqi
    return rqi


def final_score(relevance: float, rqi: float, alpha: float = 0.55) -> float:
    """Backward-compatible final score blend using relevance and RQI only."""
    alpha = _clamp01(alpha)
    score = alpha * _clamp01(relevance) + (1 - alpha) * _clamp01(rqi)
    return round(_clamp01(score), 3)


def generate_reason(signals: Mapping[str, Any], relevance: float | None = None) -> str:
    """Generate a deterministic explanation string from signal and relevance rules."""
    reasons: list[str] = []

    if _clamp01(signals.get("quality_score", 0.0)) >= 0.7:
        reasons.append("high analytical quality")
    if _clamp01(signals.get("report_validity_score", 0.0)) >= 0.7:
        reasons.append("strong report-like structure")

    if _clamp01(signals.get("methodology", signals.get("has_methodology", 0))) >= 0.6:
        reasons.append("contains methodology")

    citation_score = _clamp01(signals.get("citation_score", 0))
    if citation_score >= 0.3:
        reasons.append("strong citation support")

    if _clamp01(signals.get("consistency_score", 0)) >= 0.7:
        reasons.append("internally consistent analysis")

    if _clamp01(signals.get("recency", 0)) > 0.7:
        reasons.append("recent publication")

    if relevance is not None and _clamp01(relevance) < 0.3:
        reasons.append("low relevance to the query")

    if reasons:
        return ", ".join(reasons)
    return "limited credibility indicators"


def build_reason(signals: dict, rqi: float | None = None, relevance: float | None = None) -> str:
    """Compatibility wrapper for existing pipeline imports."""
    return generate_reason(signals, relevance=relevance)


def rank_reports(reports: list, top_k: int | None = 10, query: str | None = None) -> list:
    """Rank reports with explicit sub-scores while preserving old output keys.

    Each ranked result now includes:
    - RQI (compatibility)
    - score (compatibility, equals final_score)
    - score_breakdown (new structured score dictionary)
    - source_class (new: from source_classifier)
    - authority_prior (new: from source_classifier)
    - report_type (new: from report_classifier)
    - report_validity_score_classifier (new: from report_classifier)
    """
    ranked: list[dict[str, Any]] = []

    for report in reports:
        if not isinstance(report, dict):
            continue

        score_dict = compute_report_scores(report, query=query)
        signals = dict(report.get("signals", {}) or {})

        if report.get("relevance") is None:
            report["relevance"] = score_dict["relevance_score"]

        rqi = round(_clamp01(0.70 * score_dict["quality_score"] + 0.30 * score_dict["report_validity_score"]), 3)
        signals["quality_score"] = score_dict["quality_score"]
        signals["report_validity_score"] = score_dict["report_validity_score"]
        signals["RQI"] = rqi

        reason = generate_reason(signals, relevance=score_dict["relevance_score"])

        # NEW: Include classification outputs in final result
        ranked_item = {
            "title": report.get("title", ""),
            "url": report.get("url", ""),
            "RQI": rqi,
            "score": score_dict["final_score"],
            "reason": reason,
            "score_breakdown": score_dict,
            "source_class": signals.get("source_class", "unknown"),
            "authority_prior": signals.get("authority_prior", 0.45),
            "report_type": signals.get("report_type", "unknown"),
            "report_validity_score_classifier": signals.get("report_validity_score_classifier", 0.0),
        }
        ranked.append(ranked_item)

    ranked.sort(key=lambda item: item["score"], reverse=True)
    if top_k is None:
        return ranked
    return ranked[:max(1, top_k)]
