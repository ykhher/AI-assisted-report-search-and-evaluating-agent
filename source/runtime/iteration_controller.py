"""Failure-aware query rewrite helpers for iterative search."""

import datetime
import re
from typing import Optional


VALIDITY_THRESHOLD = 0.50       # If avg report_validity < this, might not be reports
AUTHORITY_THRESHOLD = 0.55      # If avg authority < this, sources are weak
QUALITY_THRESHOLD = 0.45        # If avg quality < this, analysis is weak

CURRENT_YEAR = datetime.datetime.now().year
FRESHNESS_YEARS = 5             # Consider results older than N years as "too old"
YEAR_THRESHOLD = CURRENT_YEAR - FRESHNESS_YEARS

MIN_RESULTS_THRESHOLD = 5       # Below this, "too_few_results"

def _extract_score(result: dict, score_key: str, default: float = 0.0) -> float:
    """Safely extract a score from result or score_breakdown."""
    if score_key in result:
        return float(result.get(score_key, default))
    
    score_breakdown = result.get("score_breakdown", {})
    return float(score_breakdown.get(score_key, default))


def _calculate_avg_score(results: list[dict], score_key: str) -> float:
    """Calculate average score across results."""
    if not results:
        return 0.0
    
    scores = [_extract_score(r, score_key) for r in results]
    valid_scores = [s for s in scores if s >= 0]
    
    if not valid_scores:
        return 0.0
    return sum(valid_scores) / len(valid_scores)


def _extract_year(result: dict) -> Optional[int]:
    """Extract year from result fields when available."""
    if "year" in result:
        year = result.get("year")
        if isinstance(year, int):
            return year
    
    for field in ("url", "title"):
        text = str(result.get(field, ""))
        match = re.search(r"\b(19|20)\d{2}\b", text)
        if match:
            return int(match.group(1))
    
    return None


def _calculate_avg_year(results: list[dict]) -> Optional[int]:
    """Calculate median year of results to assess freshness."""
    if not results:
        return None
    
    years = [_extract_year(r) for r in results if _extract_year(r) is not None]
    
    if not years:
        return None
    
    years.sort()
    return years[len(years) // 2]


def _calculate_topic_overlap(query: str, result_text: str) -> float:
    """Estimate topic overlap between query and result (title + url)."""
    if not query or not result_text:
        return 0.0
    
    query_terms = set(
        re.findall(r"\b\w{3,}\b", query.lower())
    )
    
    query_terms = {
        t for t in query_terms if t not in {
            "report", "reports", "pdf", "analysis", "industry", "market",
            "2024", "2023", "2022", "2021", "outlook", "forecast"
        }
    }
    
    if len(query_terms) < 1:
        return 0.5
    
    result_text_lower = result_text.lower()
    overlap_count = sum(1 for term in query_terms if term in result_text_lower)
    
    overlap_ratio = overlap_count / len(query_terms)
    return overlap_ratio


def diagnose_failure(results: list[dict]) -> str:
    """Diagnose why a result set is weak."""
    if len(results) < MIN_RESULTS_THRESHOLD:
        return "too_few_results"

    avg_validity = _calculate_avg_score(results, "report_validity_score")
    if avg_validity < VALIDITY_THRESHOLD:
        return "not_report_like"

    avg_authority = _calculate_avg_score(results, "authority_score")
    if avg_authority < AUTHORITY_THRESHOLD:
        return "low_authority"

    avg_year = _calculate_avg_year(results)
    if avg_year is not None and avg_year < YEAR_THRESHOLD:
        return "too_old"

    avg_quality = _calculate_avg_score(results, "quality_score")
    if avg_quality < QUALITY_THRESHOLD:
        return "weak_quality_signals"

    return "topic_drift"


FAILURE_STRATEGIES = {
    "topic_drift": {
        "add_keywords": ["analysis", "report", "industry", "market"],
        "strategy": "Broaden context with common report keywords"
    },
    "not_report_like": {
        "add_keywords": ["report", "whitepaper", "research paper", "analysis", "study"],
        "strategy": "Add explicit report-type keywords"
    },
    "low_authority": {
        "add_keywords": [".edu", ".gov", "academic", "research", "institute", "university"],
        "strategy": "Target authoritative sources (academic, government, research)"
    },
    "too_old": {
        "add_keywords": [str(CURRENT_YEAR), str(CURRENT_YEAR - 1), "latest", "recent"],
        "strategy": "Prefer recent publications"
    },
    "weak_quality_signals": {
        "add_keywords": ["methodology", "research", "findings", "analysis", "results"],
        "strategy": "Attract analytically rigorous documents"
    },
    "too_few_results": {
        "add_keywords": ["AND", "OR"],  # Placeholder for expansion logic
        "strategy": "Broaden search scope or use relaxed operators"
    }
}


def rewrite_from_failure(
    original_query: str,
    failure_type: str,
    max_additions: int = 3
) -> str:
    """Rewrite a query using the diagnosed failure type."""

    if failure_type not in FAILURE_STRATEGIES:
        return original_query

    strategy = FAILURE_STRATEGIES[failure_type]
    add_keywords = strategy.get("add_keywords", [])

    if failure_type == "too_few_results":
        restrictive = ["must", "only", "exact", "precise"]
        refined = original_query
        for term in restrictive:
            refined = re.sub(rf"\b{re.escape(term)}\b", "", refined, flags=re.IGNORECASE)
        refined = refined.strip()
        return refined if refined else original_query

    if failure_type == "low_authority":
        return f"academic research {original_query}"

    additions = add_keywords[:max_additions]
    refined_query = f"{original_query} {' '.join(additions)}"

    return refined_query.strip()

def should_iterate(
    results: list[dict],
    max_iterations: int = 3,
    current_iteration: int = 1
) -> tuple[bool, Optional[str]]:
    """Return whether to iterate again and which failure to address."""

    if current_iteration >= max_iterations:
        return (False, None)

    avg_validity = _calculate_avg_score(results, "report_validity_score")
    avg_authority = _calculate_avg_score(results, "authority_score")
    avg_quality = _calculate_avg_score(results, "quality_score")

    if avg_validity > 0.65 and avg_authority > 0.65 and avg_quality > 0.60:
        return (False, None)

    failure = diagnose_failure(results)
    return (True, failure)

def iterate_with_diagnosis(
    original_query: str,
    run_pipeline_func,
    max_iterations: int = 3,
    verbose: bool = True
) -> list[dict]:
    """Run iterative search and keep the best results seen."""

    current_query = original_query
    best_results = []
    best_score = 0.0
    
    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"[iteration_controller] Iteration {iteration}: Running '{current_query}'")

        results = run_pipeline_func(current_query, top_k=10)

        if not results:
            if verbose:
                print(f"[iteration_controller] No results found, stopping.")
            break

        avg_score = _calculate_avg_score(results, "score")

        if avg_score > best_score:
            best_results = results
            best_score = avg_score

        should_continue, failure = should_iterate(results, max_iterations, iteration)

        if not should_continue:
            if verbose:
                print(f"[iteration_controller] Iteration {iteration}: Results good, stopping.")
            break

        if failure:
            refined = rewrite_from_failure(current_query, failure)
            if verbose:
                print(f"[iteration_controller] Failure diagnosed: {failure}")
                print(f"[iteration_controller] Iteration {iteration + 1}: Refined to '{refined}'")
            current_query = refined
        else:
            break

    return best_results

if __name__ == "__main__":
    mock_results = [
        {
            "title": "AI Market Trends 2024",
            "url": "https://example.com/ai-trends",
            "score": 0.68,
            "score_breakdown": {
                "relevance_score": 0.75,
                "report_validity_score": 0.62,
                "quality_score": 0.58,
                "authority_score": 0.65,
            },
            "year": 2024,
        },
        {
            "title": "Machine Learning Blog Post",
            "url": "https://blog.example.com/ml",
            "score": 0.45,
            "score_breakdown": {
                "relevance_score": 0.70,
                "report_validity_score": 0.35,
                "quality_score": 0.32,
                "authority_score": 0.40,
            },
            "year": 2024,
        },
    ]
    
    print("=" * 80)
    print("ITERATION CONTROLLER TEST")
    print("=" * 80)
    
    print("\nTest 1: Diagnose weak results")
    failure = diagnose_failure(mock_results)
    print(f"  Failure type: {failure}")
    
    print("\nTest 2: Rewrite query from failure")
    original = "machine learning trends"
    refined = rewrite_from_failure(original, failure, max_additions=3)
    print(f"  Original: '{original}'")
    print(f"  Failure: {failure}")
    print(f"  Refined: '{refined}'")
    
    print("\nTest 3: Example rewrites for each failure type")
    test_query = "blockchain scalability"
    for failure_type in [
        "topic_drift", "not_report_like", "low_authority",
        "too_old", "weak_quality_signals", "too_few_results"
    ]:
        refined = rewrite_from_failure(test_query, failure_type)
        strategy = FAILURE_STRATEGIES[failure_type]["strategy"]
        print(f"  {failure_type:20} → '{refined}'")
        print(f"    ({strategy})")
    
    print("\nTest 4: Should iterate?")
    should_iter, next_failure = should_iterate(mock_results, max_iterations=2, current_iteration=1)
    print(f"  Should iterate: {should_iter}")
    if next_failure:
        print(f"  Next failure to address: {next_failure}")
    
    print("\n" + "=" * 80)
