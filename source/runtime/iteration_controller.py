"""Failure-aware query rewrite helpers for iterative search."""

from __future__ import annotations

import datetime
import re


VALIDITY_THRESHOLD = 0.50
AUTHORITY_THRESHOLD = 0.55
QUALITY_THRESHOLD = 0.45

CURRENT_YEAR = datetime.datetime.now().year
FRESHNESS_YEARS = 5
YEAR_THRESHOLD = CURRENT_YEAR - FRESHNESS_YEARS

MIN_RESULTS_THRESHOLD = 5


def _extract_score(result: dict, score_key: str, default: float = 0.0) -> float:
    """Safely extract a score from the top level or score_breakdown."""
    try:
        if score_key in result:
            return float(result.get(score_key, default))
        breakdown = result.get("score_breakdown", {})
        if isinstance(breakdown, dict):
            return float(breakdown.get(score_key, default))
    except (TypeError, ValueError):
        return default
    return default


def _average_score(results: list[dict], score_key: str) -> float:
    """Calculate an average score across results."""
    if not results:
        return 0.0
    scores = [_extract_score(result, score_key) for result in results]
    valid_scores = [score for score in scores if score >= 0]
    return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0


def _extract_year(result: dict) -> int | None:
    """Extract year from result fields when available."""
    year = result.get("year")
    if isinstance(year, int):
        return year

    for field in ("url", "title"):
        text = str(result.get(field, ""))
        match = re.search(r"\b(?:19|20)\d{2}\b", text)
        if match:
            return int(match.group(0))
    return None


def _median_year(results: list[dict]) -> int | None:
    """Calculate the median year of results to assess freshness."""
    years = [_extract_year(result) for result in results]
    valid_years = sorted(year for year in years if year is not None)
    if not valid_years:
        return None
    return valid_years[len(valid_years) // 2]


def diagnose_failure(results: list[dict]) -> str:
    """Diagnose why a result set looks weak."""
    if len(results) < MIN_RESULTS_THRESHOLD:
        return "too_few_results"

    if _average_score(results, "report_validity_score") < VALIDITY_THRESHOLD:
        return "not_report_like"

    if _average_score(results, "authority_score") < AUTHORITY_THRESHOLD:
        return "low_authority"

    avg_year = _median_year(results)
    if avg_year is not None and avg_year < YEAR_THRESHOLD:
        return "too_old"

    if _average_score(results, "quality_score") < QUALITY_THRESHOLD:
        return "weak_quality_signals"

    return "topic_drift"


FAILURE_STRATEGIES = {
    "topic_drift": {
        "add_keywords": ["analysis", "report", "industry", "market"],
        "strategy": "Broaden context with common report keywords",
    },
    "not_report_like": {
        "add_keywords": ["report", "whitepaper", "research paper", "analysis", "study"],
        "strategy": "Add explicit report-type keywords",
    },
    "low_authority": {
        "add_keywords": [".edu", ".gov", "academic", "research", "institute", "university"],
        "strategy": "Target authoritative sources (academic, government, research)",
    },
    "too_old": {
        "add_keywords": [str(CURRENT_YEAR), str(CURRENT_YEAR - 1), "latest", "recent"],
        "strategy": "Prefer recent publications",
    },
    "weak_quality_signals": {
        "add_keywords": ["methodology", "research", "findings", "analysis", "results"],
        "strategy": "Attract analytically rigorous documents",
    },
    "too_few_results": {
        "add_keywords": ["AND", "OR"],
        "strategy": "Broaden search scope or use relaxed operators",
    },
}


def rewrite_from_failure(original_query: str, failure_type: str, max_additions: int = 3) -> str:
    """Rewrite a query using the diagnosed failure type."""
    strategy = FAILURE_STRATEGIES.get(failure_type)
    if not strategy:
        return original_query

    if failure_type == "too_few_results":
        refined = original_query
        for term in ("must", "only", "exact", "precise"):
            refined = re.sub(rf"\b{re.escape(term)}\b", "", refined, flags=re.IGNORECASE)
        refined = refined.strip()
        return refined or original_query

    if failure_type == "low_authority":
        return f"academic research {original_query}".strip()

    additions = strategy.get("add_keywords", [])[:max_additions]
    return f"{original_query} {' '.join(additions)}".strip()


def should_iterate(
    results: list[dict],
    max_iterations: int = 3,
    current_iteration: int = 1,
) -> tuple[bool, str | None]:
    """Return whether to iterate again and which failure to address."""
    if current_iteration >= max_iterations:
        return False, None

    if (
        _average_score(results, "report_validity_score") > 0.65
        and _average_score(results, "authority_score") > 0.65
        and _average_score(results, "quality_score") > 0.60
    ):
        return False, None

    return True, diagnose_failure(results)


def iterate_with_diagnosis(
    original_query: str,
    run_pipeline_func,
    max_iterations: int = 3,
    verbose: bool = True,
) -> list[dict]:
    """Run iterative search and keep the best results seen."""
    current_query = original_query
    best_results: list[dict] = []
    best_score = 0.0

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"[iteration_controller] Iteration {iteration}: Running '{current_query}'")

        results = run_pipeline_func(current_query, top_k=10)
        if not results:
            if verbose:
                print("[iteration_controller] No results found, stopping.")
            break

        avg_score = _average_score(results, "score")
        if avg_score > best_score:
            best_results = results
            best_score = avg_score

        should_continue, failure = should_iterate(results, max_iterations, iteration)
        if not should_continue:
            if verbose:
                print(f"[iteration_controller] Iteration {iteration}: Results good, stopping.")
            break

        if failure:
            current_query = rewrite_from_failure(current_query, failure)
            if verbose:
                print(f"[iteration_controller] Failure diagnosed: {failure}")
                print(f"[iteration_controller] Iteration {iteration + 1}: Refined to '{current_query}'")

    return best_results
