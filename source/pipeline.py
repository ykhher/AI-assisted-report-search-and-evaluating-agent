"""End-to-end report search and ranking pipeline."""

from __future__ import annotations

import json
import re
from typing import Any

from extractor import extract_signals, is_report, source_score
from filter import filter_results
from scoring import compute_relevance_score, compute_rqi, final_score, generate_reason, rank_reports
from search import expand_query, search_reports
from source_classifier import classify_source
from report_classifier import classify_report_type

_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


def _infer_year(doc: dict[str, Any]) -> int | None:
    """Infer a publication year from available metadata fields."""
    direct_year = doc.get("year")
    if isinstance(direct_year, int):
        return direct_year

    for field in ("date", "title", "url", "snippet"):
        match = _YEAR_PATTERN.search(str(doc.get(field, "")))
        if match:
            return int(match.group(0))
    return None


def _get_report_text(doc: dict[str, Any]) -> str:
    """Return text content from a search result."""
    return str(doc.get("text") or doc.get("snippet") or "").strip()


def run_pipeline(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Run search, filter report-like results, and return ranked items."""
    expanded_query = expand_query(query)
    raw_results = search_reports(query, count=max(20, top_k * 2))
    print(f"[pipeline] Expanded query: {expanded_query}")
    print(f"[pipeline] Retrieved {len(raw_results)} candidate result(s)")

    coarse_results = filter_results(
        raw_results,
        min_score=0.0,
        keywords=["report", "analysis", "outlook", "benchmark", "survey", "review"],
    )
    candidates = coarse_results if coarse_results else raw_results

    prepared_reports: list[dict[str, Any]] = []

    for doc in candidates:
        if not isinstance(doc, dict):
            continue

        text = _get_report_text(doc)
        if not text:
            print(f"[pipeline] Skipping result with missing text: {doc.get('title', 'untitled')}")
            continue

        combined_text = f"{doc.get('title', '')} {text}".strip()
        metadata = {
            "is_pdf": bool(doc.get("is_pdf", False)),
            "source": doc.get("source", ""),
            "year": _infer_year(doc),
        }

        report_like_terms = ("report", "outlook", "benchmark", "survey", "analysis", "whitepaper")
        looks_like_report = any(term in combined_text.lower() for term in report_like_terms)
        if not (is_report(combined_text, metadata) or (metadata["is_pdf"] and looks_like_report)):
            print(f"[pipeline] Filtered out (not a valid report): {doc.get('title', 'untitled')}")
            continue

        signals = extract_signals(combined_text, metadata)
        signals["source_name"] = str(metadata.get("source", ""))
        signals["source"] = source_score(metadata.get("source", ""), combined_text)

        source_url = str(doc.get("url", ""))
        source_result = classify_source(
            url=source_url,
            title=doc.get("title", ""),
            text=text[:500] if text else ""
        )
        signals["source_class"] = source_result.get("source_class", "unknown")
        signals["authority_prior"] = source_result.get("authority_prior", 0.45)

        report_result = classify_report_type(
            title=doc.get("title", ""),
            text=text,
            metadata=metadata
        )
        signals["report_type"] = report_result.get("report_type", "unknown")
        signals["report_validity_score_classifier"] = report_result.get("report_validity_score", 0.0)

        relevance = compute_relevance_score(query, combined_text)
        rqi = compute_rqi(signals)
        score = final_score(relevance, rqi)
        reason = generate_reason(signals)

        prepared_reports.append({
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "relevance": relevance,
            "signals": signals,
            "RQI": rqi,
            "score": score,
            "reason": reason,
        })

    if not prepared_reports:
        print("[pipeline] No valid reports found.")
        return []

    ranked_results = rank_reports(prepared_reports, top_k=top_k)
    return ranked_results


def pipeline(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Backward-compatible wrapper around `run_pipeline`."""
    return run_pipeline(query, top_k=top_k)


def print_results(results: list[dict]) -> None:
    """Pretty-print ranked results to stdout as formatted JSON."""
    if not results:
        print("No valid reports found.")
        return

    print("\n" + "=" * 60)
    print("RANKED REPORT RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print("=" * 60 + "\n")
