"""Reflection helpers for the report agent."""

from __future__ import annotations

from datetime import datetime
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.agent_state import AgentState, CandidateRecord


MIN_USABLE_CANDIDATES = 3
GOOD_TOP_SCORE = 0.70
GOOD_AVG_SCORE = 0.62
MIN_RELEVANCE = 0.35
MIN_VALIDITY = 0.50
MIN_AUTHORITY = 0.55
MIN_QUALITY = 0.45
MAX_RESULT_AGE_YEARS = 5
FETCH_FAILURE_DOMINANCE_RATIO = 0.60


def _to_payload(candidate: CandidateRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(candidate, CandidateRecord):
        return candidate.to_payload()
    if isinstance(candidate, dict):
        return dict(candidate)
    return {}


def _score(candidate: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        if key in candidate:
            return float(candidate.get(key, default))
        breakdown = candidate.get("score_breakdown", {})
        if isinstance(breakdown, dict):
            return float(breakdown.get(key, default))
    except (TypeError, ValueError):
        return default
    return default


def _year(candidate: dict[str, Any]) -> int | None:
    value = candidate.get("year")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median_year(years: list[int]) -> int | None:
    if not years:
        return None
    years = sorted(years)
    return years[len(years) // 2]


def _resolve_candidates(
    state: AgentState,
    candidates: list[dict[str, Any] | CandidateRecord] | None = None,
) -> list[dict[str, Any]]:
    items = candidates if candidates is not None else state.candidates
    return [payload for payload in (_to_payload(item) for item in items) if payload]


def _usable_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in candidates
        if str(candidate.get("status", "")).strip().lower() not in {"failed", "filtered_out"}
    ]


def evaluate_progress(
    state: AgentState,
    candidates: list[dict[str, Any] | CandidateRecord] | None = None,
) -> dict[str, Any]:
    """Summarize the current run in a way the controller can use."""
    resolved = _resolve_candidates(state, candidates)
    usable = _usable_candidates(resolved)

    final_scores = [_score(candidate, "final_score", _score(candidate, "score")) for candidate in usable]
    years = [year for year in (_year(candidate) for candidate in usable) if year is not None]
    total_seen = len(state.visited_urls)
    failed_urls = len(state.failed_urls)
    failure_ratio = failed_urls / total_seen if total_seen else 0.0

    progress = {
        "total_candidates": len(resolved),
        "usable_candidates": len(usable),
        "filtered_out_count": len(state.filtered_out),
        "visited_url_count": total_seen,
        "failed_url_count": failed_urls,
        "fetch_failure_ratio": round(failure_ratio, 3),
        "avg_relevance": round(_average([_score(candidate, "relevance_score") for candidate in usable]), 3),
        "avg_report_validity": round(_average([_score(candidate, "report_validity_score") for candidate in usable]), 3),
        "avg_authority": round(_average([_score(candidate, "authority_score") for candidate in usable]), 3),
        "avg_quality": round(_average([_score(candidate, "quality_score") for candidate in usable]), 3),
        "avg_final_score": round(_average(final_scores), 3),
        "top_final_score": round(max(final_scores), 3) if final_scores else 0.0,
        "median_year": _median_year(years),
        "current_step": state.current_step,
        "stop_reason": state.stop_reason,
        "failure_history": list(state.failure_history),
    }
    progress["diagnosis"] = diagnose_failure(state, resolved)
    return progress


def diagnose_failure(
    state: AgentState,
    candidates: list[dict[str, Any] | CandidateRecord] | None = None,
) -> str | None:
    """Return the main reason progress looks weak."""
    resolved = _resolve_candidates(state, candidates)
    usable = _usable_candidates(resolved)

    total_seen = max(len(state.visited_urls), len(resolved))
    failed_urls = len(state.failed_urls)
    failure_ratio = failed_urls / total_seen if total_seen else 0.0
    if total_seen and failure_ratio >= FETCH_FAILURE_DOMINANCE_RATIO:
        return "fetch_failures_dominant"
    if len(usable) < MIN_USABLE_CANDIDATES:
        return "too_few_results"
    if _average([_score(candidate, "relevance_score") for candidate in usable]) < MIN_RELEVANCE:
        return "topic_drift"
    if _average([_score(candidate, "report_validity_score") for candidate in usable]) < MIN_VALIDITY:
        return "not_report_like"
    if _average([_score(candidate, "authority_score") for candidate in usable]) < MIN_AUTHORITY:
        return "low_authority"

    median_year = _median_year([year for year in (_year(candidate) for candidate in usable) if year is not None])
    if median_year is not None and median_year < datetime.now().year - MAX_RESULT_AGE_YEARS:
        return "too_old"

    quality = _average([_score(candidate, "quality_score") for candidate in usable])
    evidence_hits = 0
    for candidate in usable:
        signals = candidate.get("signals", {})
        if not isinstance(signals, dict):
            continue
        has_methodology = bool(signals.get("methodology")) or bool(signals.get("parsed_has_methodology"))
        has_references = bool(signals.get("citation_score")) or bool(signals.get("parsed_has_references"))
        if has_methodology or has_references:
            evidence_hits += 1

    evidence_ratio = evidence_hits / len(usable) if usable else 0.0
    if quality < MIN_QUALITY or evidence_ratio < 0.35:
        return "weak_quality_signals"
    return None


def should_stop(
    state: AgentState,
    candidates: list[dict[str, Any] | CandidateRecord] | None = None,
) -> bool:
    """Stop when results look good enough.

    The controller owns the iteration budget, so repeated diagnoses should
    trigger replanning until that budget is exhausted rather than stopping
    reflection early.
    """
    if state.stop_reason:
        return True

    progress = evaluate_progress(state, candidates)
    return (
        progress["usable_candidates"] >= MIN_USABLE_CANDIDATES
        and progress["top_final_score"] >= GOOD_TOP_SCORE
        and progress["avg_final_score"] >= GOOD_AVG_SCORE
    )


def summarize_progress(
    state: AgentState,
    candidates: list[dict[str, Any] | CandidateRecord] | None = None,
) -> str:
    """Return one short line about the current run."""
    progress = evaluate_progress(state, candidates)
    parts = [
        f"step={progress['current_step']}",
        f"usable_candidates={progress['usable_candidates']}",
        f"avg_final_score={progress['avg_final_score']:.3f}",
        f"top_final_score={progress['top_final_score']:.3f}",
        f"failed_urls={progress['failed_url_count']}",
    ]
    if progress.get("diagnosis"):
        parts.append(f"diagnosis={progress['diagnosis']}")
    if progress.get("median_year") is not None:
        parts.append(f"median_year={progress['median_year']}")
    return ", ".join(parts)
