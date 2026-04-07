"""Cost-aware iterative agent layer for the report ranking system."""

from __future__ import annotations

import re
import sys
from typing import Any
from urllib.parse import urlparse

from extractor import extract_signals, is_report, source_score
from filter import filter_results
from scoring import compute_rqi, final_score, generate_reason, rank_reports
from search import search_once

_BASE_HINTS = ["pdf", "2024", "industry", "analysis", "forecast"]
_ITERATION_HINTS = [
    ["benchmark", "survey"],
    ["outlook", "whitepaper"],
]
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


def refine_query(query: str) -> str:
    """Expand a user query with report-friendly search terms."""
    cleaned = " ".join(str(query or "").strip().split())
    if not cleaned:
        cleaned = "industry report"

    lower = cleaned.lower()
    missing_terms = [term for term in _BASE_HINTS if term not in lower]
    return f"{cleaned} {' '.join(missing_terms)}".strip()


def evaluate_results(results: list) -> str:
    """Return `low_quality` when the average top-3 score is below 0.5."""
    if not results:
        return "low_quality"

    top_results = results[:3]
    scores = []
    for item in top_results:
        try:
            score = float(item.get("score", 0))
        except (AttributeError, TypeError, ValueError):
            score = 0.0
        scores.append(max(0.0, min(score, 1.0)))

    average_score = sum(scores) / len(scores) if scores else 0.0
    return "low_quality" if average_score < 0.5 else "ok"


def _average_top_score(results: list[dict[str, Any]]) -> float:
    """Compute the average score of the top three results."""
    if not results:
        return 0.0
    top_results = results[:3]
    scores = [float(item.get("score", 0) or 0) for item in top_results]
    return sum(scores) / len(scores)


def _strengthen_query(query: str, iteration: int) -> str:
    """Add extra deterministic hints when a previous iteration was weak."""
    refined = refine_query(query)
    extra_terms = _ITERATION_HINTS[min(iteration, len(_ITERATION_HINTS) - 1)]
    lower = refined.lower()
    missing_terms = [term for term in extra_terms if term not in lower]
    return f"{refined} {' '.join(missing_terms)}".strip()


def _infer_year(doc: dict[str, Any]) -> int | None:
    """Infer a year from the cached result fields."""
    direct_year = doc.get("year")
    if isinstance(direct_year, int):
        return direct_year

    for field in ("date", "title", "url", "snippet"):
        match = _YEAR_PATTERN.search(str(doc.get(field, "")))
        if match:
            return int(match.group(0))
    return None


def _topic_terms(query: str) -> set[str]:
    """Extract the meaningful topic words from a user query."""
    generic_terms = {
        "pdf", "report", "reports", "2024", "2025", "industry", "analysis",
        "market", "forecast", "outlook", "cagr", "projection", "research",
        "size", "revenue", "global",
    }
    terms = {
        word for word in re.findall(r"[a-z0-9]+", query.lower())
        if len(word) > 2 and word not in generic_terms
    }
    return terms or {word for word in re.findall(r"[a-z0-9]+", query.lower()) if len(word) > 2}


def _compute_relevance(query: str, text: str) -> float:
    """Compute simple keyword-overlap relevance normalized to [0,1]."""
    query_words = _topic_terms(query)
    if not query_words:
        return 0.0

    lowered_text = text.lower()
    text_words = set(re.findall(r"[a-z0-9]+", lowered_text))
    overlap_ratio = len(query_words & text_words) / len(query_words)
    exact_phrase_boost = 0.15 if query.lower().strip() in lowered_text else 0.0
    return round(min(overlap_ratio + exact_phrase_boost, 1.0), 3)


def _score_cached_results(cached_results: list[dict[str, Any]], refined_query: str, iteration: int) -> list:
    """Process cached API results locally without making any additional API calls."""
    topic_terms = _topic_terms(refined_query)
    query_words = [word for word in re.findall(r"[a-z0-9]+", refined_query.lower()) if len(word) > 2]
    candidates = filter_results(cached_results, min_score=0.0, keywords=query_words[:6]) or cached_results

    prepared_reports: list[dict[str, Any]] = []
    for doc in candidates:
        if not isinstance(doc, dict):
            continue

        text = str(doc.get("text") or doc.get("snippet") or "").strip()
        if not text:
            continue

        combined_text = f"{doc.get('title', '')} {text}".strip()
        combined_text_lower = combined_text.lower()
        topic_overlap = sum(1 for term in topic_terms if term in combined_text_lower)
        if topic_terms and topic_overlap == 0:
            continue

        source = doc.get("source") or urlparse(str(doc.get("url", ""))).netloc
        metadata = {
            "is_pdf": bool(doc.get("is_pdf", False) or str(doc.get("url", "")).lower().endswith(".pdf")),
            "source": source,
            "year": _infer_year(doc),
        }

        report_like_terms = (
            "report", "outlook", "benchmark", "survey", "analysis", "whitepaper",
            "market size", "industry report",
        )
        looks_like_report = any(term in combined_text_lower for term in report_like_terms)
        if not (is_report(combined_text, metadata) or (metadata["is_pdf"] and (looks_like_report or topic_overlap > 0))):
            continue

        signals = extract_signals(combined_text, metadata)
        signals["source"] = source_score(source)
        signals["_text"] = combined_text
        relevance = _compute_relevance(refined_query, combined_text)
        relevance = min(relevance + 0.05 * min(topic_overlap, 3), 1.0)
        if iteration > 0 and signals.get("methodology"):
            relevance = min(relevance + 0.05, 1.0)

        rqi = compute_rqi(signals, combined_text)
        score = final_score(relevance, rqi)
        reason = generate_reason(signals, relevance=relevance)

        prepared_reports.append({
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "relevance": relevance,
            "signals": signals,
            "RQI": rqi,
            "score": score,
            "reason": reason,
        })

    ranked_results = rank_reports(prepared_reports)
    return ranked_results


def agent_pipeline(query: str, max_iters: int = 2) -> list:
    """Run the agent loop while calling the real search API only once."""
    cached_results = search_once(query)
    if not cached_results:
        print("[agent] Search API failed or returned no results.")
        return []

    best_results: list = []
    best_score = -1.0

    for iteration in range(max(1, max_iters)):
        refined_query = _strengthen_query(query, iteration)
        print(f"[agent] Iteration {iteration + 1} query: {refined_query}")
        print(f"[agent] Reusing {len(cached_results)} cached API result(s)")

        results = _score_cached_results(cached_results, refined_query, iteration)
        if not results:
            print("[agent] No valid reports found in cached results.")
            return []

        avg_score = _average_top_score(results)
        print(f"[agent] Average top-3 score: {avg_score:.3f}")

        if avg_score > best_score:
            best_score = avg_score
            best_results = results

        quality = evaluate_results(results)
        print(f"[agent] Evaluation: {quality}")

        if quality == "ok":
            print("[agent] Result quality is sufficient; stopping early.")
            return results

    return best_results


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip() or "AI market report"
    results = agent_pipeline(query)
    for r in results:
        print(r)
