"""Iterative agent layer for report ranking."""

from __future__ import annotations

import importlib.util
import re
import sys
import time
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from extractor import extract_signals, is_report, source_score
from filter import filter_results
from iteration_controller import diagnose_failure, rewrite_from_failure
from local_qwen import get_local_qwen_status, rewrite_search_query
from scoring import compute_rqi, final_score, generate_reason, rank_reports
from search import search_once
from exporter import export_to_csv, export_to_json
from schemas import BatchResults, RankedReport, ScoreBreakdown

_BASE_HINTS = ["pdf", "2024", "industry", "analysis", "forecast"]
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_DEFAULT_FETCH_TOP_N = 8


def _load_optional_function(module_name: str, function_name: str):
    """Load an optional function from normal imports or fallback file paths."""
    try:
        module = __import__(module_name, fromlist=[function_name])
        return getattr(module, function_name, None)
    except Exception:
        pass

    candidate_paths = [
        Path(__file__).with_name(f"{module_name}.py"),
        Path(__file__).parent / "full agent" / f"{module_name}.py",
    ]

    for file_path in candidate_paths:
        if not file_path.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"{module_name}_dynamic", file_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, function_name, None)
        except Exception:
            continue

    return None


_FETCH_DOCUMENT = _load_optional_function("document_fetcher", "fetch_document")
_PARSE_REPORT_TEXT = _load_optional_function("text_parser", "parse_report_text")


def refine_query(query: str) -> str:
    """Expand a user query with report-friendly terms, optionally using the local Qwen model."""
    cleaned = " ".join(str(query or "").strip().split())
    if not cleaned:
        cleaned = "industry report"

    rewritten = rewrite_search_query(cleaned)
    if rewritten:
        cleaned = rewritten

    lower = cleaned.lower()
    missing_terms = [term for term in _BASE_HINTS if term not in lower]
    return f"{cleaned} {' '.join(missing_terms)}".strip()


def evaluate_results(results: list) -> str:
    """Return 'ok' when top results look good, otherwise a failure type."""
    if not results:
        return "too_few_results"

    top_results = results[:3]
    scores = []
    for item in top_results:
        try:
            score = float(item.get("score", 0))
        except (AttributeError, TypeError, ValueError):
            score = 0.0
        scores.append(max(0.0, min(score, 1.0)))

    average_score = sum(scores) / len(scores) if scores else 0.0

    if average_score < 0.5:
        failure = diagnose_failure(results)
        return failure

    return "ok"


def _average_top_score(results: list[dict[str, Any]]) -> float:
    """Compute the average score of the top three results."""
    if not results:
        return 0.0
    top_results = results[:3]
    scores = [float(item.get("score", 0) or 0) for item in top_results]
    return sum(scores) / len(scores)


def _strengthen_query(query: str, failure_type: str | None) -> str:
    """Refine the query using the last failure reason when available."""
    refined_base = refine_query(query)

    if not failure_type or failure_type == "ok":
        return refined_base

    try:
        rewritten = rewrite_from_failure(query, failure_type, max_additions=3)
        if rewritten != query:
            return rewritten
    except Exception as exc:
        print(f"[agent] rewrite_from_failure failed for {failure_type}: {exc}")

    return refined_base


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


def _score_cached_results(
    cached_results: list[dict[str, Any]],
    refined_query: str,
    iteration: int,
    top_k: int = 10,
    fetch_top_n: int = _DEFAULT_FETCH_TOP_N,
) -> list:
    """Process cached API results locally without making any additional API calls."""
    topic_terms = _topic_terms(refined_query)
    query_words = [word for word in re.findall(r"[a-z0-9]+", refined_query.lower()) if len(word) > 2]
    candidates = filter_results(cached_results, min_score=0.0, keywords=query_words[:6]) or cached_results

    # Pre-select likely report candidates (snippet-level) to cap expensive fetches.
    likely_reports: list[dict[str, Any]] = []
    for doc in candidates:
        if not isinstance(doc, dict):
            continue
        snippet_text = str(doc.get("text") or doc.get("snippet") or "").strip()
        if not snippet_text:
            continue
        combined = f"{doc.get('title', '')} {snippet_text}".lower()
        overlap = sum(1 for term in topic_terms if term in combined)
        if overlap == 0 and topic_terms:
            continue
        likely_reports.append(doc)

    to_fetch = {
        id(doc): doc
        for doc in likely_reports[: max(0, min(fetch_top_n, len(likely_reports)))]
    }

    prepared_reports: list[dict[str, Any]] = []
    for doc in candidates:
        if not isinstance(doc, dict):
            continue

        text = str(doc.get("text") or doc.get("snippet") or "").strip()
        fetch_meta: dict[str, Any] = {}
        parsed_meta: dict[str, Any] = {}

        if id(doc) in to_fetch and _FETCH_DOCUMENT is not None:
            try:
                fetched = _FETCH_DOCUMENT(str(doc.get("url", "")))
                if isinstance(fetched, dict):
                    fetch_meta = fetched
                    fetched_text = str(fetched.get("raw_text") or "").strip()
                    if fetched_text:
                        text = fetched_text

                    if _PARSE_REPORT_TEXT is not None and fetched_text:
                        try:
                            parsed = _PARSE_REPORT_TEXT(fetched_text)
                            if isinstance(parsed, dict):
                                parsed_meta = parsed
                        except Exception as exc:
                            print(f"[agent] parse_report_text failed for {doc.get('url', '')}: {exc}")
            except Exception as exc:
                print(f"[agent] fetch_document failed for {doc.get('url', '')}: {exc}")

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
        signals["source_name"] = str(source)
        signals["source"] = source_score(source, combined_text)
        signals["_text"] = combined_text
        if fetch_meta:
            signals["fetched_status"] = str(fetch_meta.get("status", ""))
            signals["fetched_content_type"] = str(fetch_meta.get("content_type", ""))
            signals["fetch_error"] = str(fetch_meta.get("error", ""))
        if parsed_meta:
            sections = parsed_meta.get("sections", {})
            stats = parsed_meta.get("stats", {})
            if isinstance(stats, dict):
                signals["parsed_word_count"] = int(stats.get("word_count", 0) or 0)
                signals["parsed_has_methodology"] = bool(stats.get("has_methodology", False))
                signals["parsed_has_references"] = bool(stats.get("has_references", False))
                signals["parsed_has_statistics_language"] = bool(stats.get("has_statistics_language", False))
            if isinstance(sections, dict):
                signals["parsed_section_lengths"] = {
                    key: len(str(value).split())
                    for key, value in sections.items()
                    if value
                }
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

    ranked_results = rank_reports(prepared_reports, top_k=top_k)
    return ranked_results


def _as_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely with a fallback default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ranked_dict_to_schema(query: str, item: dict[str, Any], index: int) -> RankedReport:
    """Convert one ranked dictionary result into a RankedReport schema object."""
    score_breakdown_data = item.get("score_breakdown") or {}
    score_breakdown = ScoreBreakdown(
        relevance_score=_as_float(score_breakdown_data.get("relevance_score", item.get("relevance", 0.0))),
        report_validity_score=_as_float(score_breakdown_data.get("report_validity_score", 0.0)),
        quality_score=_as_float(score_breakdown_data.get("quality_score", 0.0)),
        authority_score=_as_float(score_breakdown_data.get("authority_score", 0.0)),
        final_score=_as_float(score_breakdown_data.get("final_score", item.get("score", 0.0))),
    )

    source = str(item.get("source") or urlparse(str(item.get("url", ""))).netloc)
    warnings = item.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = [] if warnings in (None, "") else [str(warnings)]

    return RankedReport(
        query=str(query),
        title=str(item.get("title", "")),
        url=str(item.get("url", "")),
        year=item.get("year") if isinstance(item.get("year"), int) else None,
        source=source,
        source_class=str(item.get("source_class", "unknown")),
        authority_prior=_as_float(item.get("authority_prior", 0.45), 0.45),
        report_type=str(item.get("report_type", "unknown")),
        report_validity_score_classifier=_as_float(item.get("report_validity_score_classifier", 0.0)),
        relevance_score=score_breakdown.relevance_score,
        report_validity_score=score_breakdown.report_validity_score,
        quality_score=score_breakdown.quality_score,
        authority_score=score_breakdown.authority_score,
        final_score=_as_float(item.get("score", score_breakdown.final_score)),
        score_breakdown=score_breakdown,
        top_signals=item.get("signals") if isinstance(item.get("signals"), dict) else {},
        reasoning=str(item.get("reason", "")),
        warnings=warnings,
        index=index,
    )


def agent_pipeline(
    query: str,
    max_iters: int = 2,
    top_k: int = 10,
    output_format: Literal["list", "dict", "object"] = "dict",
    export_results: bool = False,
    export_dir: str | Path = "outputs",
) -> list | dict[str, Any] | BatchResults:
    """Run iterative ranking and return list, dict, or BatchResults output."""
    started_at = time.perf_counter()
    qwen_status = get_local_qwen_status()
    if qwen_status["available"]:
        print(f"[agent] Local Qwen enabled: {qwen_status['model_path']}")

    search_query = refine_query(query)
    print(f"[agent] Initial search query: {search_query}")
    cached_results = search_once(search_query, count=max(20, top_k * 2))
    if not cached_results:
        print("[agent] Search API failed or returned no results.")
        if output_format == "list":
            return []
        empty_batch = BatchResults(query=query, results=[], total_count=0, returned_count=0)
        if output_format == "object":
            return empty_batch
        return empty_batch.to_dict()

    best_results: list = []
    best_score = -1.0
    failure_type: str | None = None
    iterations_used = 0

    for iteration in range(max(1, max_iters)):
        iterations_used = iteration + 1
        if iteration == 0:
            refined_query = refine_query(query)
        else:
            refined_query = _strengthen_query(query, failure_type)
            if failure_type and failure_type != "ok":
                print(f"[agent] Iteration {iteration + 1}: Diagnosed '{failure_type}', refined query for targeted search")
        
        print(f"[agent] Iteration {iteration + 1} refined query: {refined_query}")
        print(f"[agent] Reusing {len(cached_results)} cached API result(s) (no additional API calls)")

        results = _score_cached_results(cached_results, refined_query, iteration, top_k=top_k)
        if not results:
            print("[agent] No valid reports found in cached results.")
            final_results: list[dict[str, Any]] = []
            if output_format == "list" and not export_results:
                return final_results
            empty_batch = BatchResults(
                query=query,
                results=[],
                total_count=len(cached_results),
                returned_count=0,
                iteration_count=iterations_used,
                failure_type="too_few_results",
                processing_time_ms=round((time.perf_counter() - started_at) * 1000.0, 3),
            )
            if output_format == "object":
                return empty_batch
            return empty_batch.to_dict()

        avg_score = _average_top_score(results)
        print(f"[agent] Average top-3 score: {avg_score:.3f}")

        if avg_score > best_score:
            best_score = avg_score
            best_results = results

        quality = evaluate_results(results)
        if quality == "ok":
            print(f"[agent] Evaluation: PASS (avg score {avg_score:.3f} acceptable)")
            print("[agent] Result quality is sufficient; stopping early.")
            best_results = results
            failure_type = "ok"
            break
        else:
            failure_type = quality
            print(f"[agent] Evaluation: FAIL (failure type: {failure_type})")

    print(f"[agent] Completed {max_iters} iteration(s); returning best results")
    final_results = best_results[:top_k]

    if output_format == "list" and not export_results:
        return final_results

    structured_results = [
        _ranked_dict_to_schema(query, item, index=i + 1)
        for i, item in enumerate(final_results)
    ]
    batch = BatchResults(
        query=query,
        results=structured_results,
        total_count=len(cached_results),
        returned_count=len(structured_results),
        iteration_count=iterations_used,
        failure_type=failure_type,
        processing_time_ms=round((time.perf_counter() - started_at) * 1000.0, 3),
    )

    if export_results:
        output_path = Path(export_dir)
        export_to_json(batch, output_path / "ranked_reports.json", indent=2)
        export_to_csv(batch.results, output_path / "ranked_reports.csv")

    if output_format == "list":
        return final_results
    if output_format == "object":
        return batch
    return batch.to_dict()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip() or "AI market report"
    results = agent_pipeline(query)
    for r in results:
        print(r)
