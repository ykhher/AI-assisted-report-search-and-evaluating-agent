"""Simple tool registry for the report agent."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.classifier.report_classifier import classify_report_type
from source.classifier.source_classifier import classify_source
from source.extractor import extract_signals
from source.fetching.document_fetcher import fetch_document
from source.fetching.text_parser import parse_report_text
from source.scoring import compute_report_scores, rank_reports
from source.search import search_reports
from source.verification import attach_verification_notes, extract_key_claims


ToolCallable = Callable[..., Any]


def search(query: str, count: int = 10) -> list[dict[str, Any]]:
    """Use the normal API-first search path with local fallback."""
    return search_reports(query, count=count, use_api=True)


def score_candidates(candidates: list[dict[str, Any]], query: str | None = None) -> list[dict[str, Any]]:
    """Add score details to each candidate."""
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        score_breakdown = compute_report_scores(candidate, query=query)
        enriched = dict(candidate)
        enriched["score_breakdown"] = score_breakdown
        enriched["score"] = score_breakdown["final_score"]
        scored.append(enriched)
    return scored


def rank_candidates(
    candidates: list[dict[str, Any]],
    top_k: int | None = 10,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """Rank candidates with the current scoring code."""
    return rank_reports(candidates, top_k=top_k, query=query)


TOOL_REGISTRY: dict[str, ToolCallable] = {
    "search": search,
    "fetch_document": fetch_document,
    "parse_document": parse_report_text,
    "classify_source": classify_source,
    "classify_report_type": classify_report_type,
    "extract_signals": extract_signals,
    "score_candidates": score_candidates,
    "rank_candidates": rank_candidates,
    "extract_key_claims": extract_key_claims,
    "attach_verification_notes": attach_verification_notes,
}


def get_tool_registry() -> dict[str, ToolCallable]:
    """Return a copy so callers do not mutate the shared map."""
    return dict(TOOL_REGISTRY)


def get_tool(name: str) -> ToolCallable:
    """Look up one tool by name."""
    name = str(name).strip()
    if name not in TOOL_REGISTRY:
        available = ", ".join(sorted(TOOL_REGISTRY))
        raise KeyError(f"Unknown tool '{name}'. Available tools: {available}")
    return TOOL_REGISTRY[name]


def list_tools() -> list[str]:
    return sorted(TOOL_REGISTRY)
