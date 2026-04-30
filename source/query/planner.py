"""Rule-based task planner for the report discovery agent.
so the agent can interpret a raw user request as a structured task with priorities.

Example:
    >>> make_plan("Find the most credible enterprise AI adoption benchmark for 2025")
    {
        "task_type": "report_discovery",
        "topic": "enterprise ai adoption",
        "year_constraint": 2025,
        "quality_priority": True,
        "preferred_report_types": ["report", "benchmark"],
        ...
    }
"""

from __future__ import annotations

from typing import Any

from source.query.handler import _base_topic, _extract_years, _normalize_query_text

_QUALITY_HINTS = {
    "best",
    "credible",
    "credibility",
    "high quality",
    "quality",
    "rigorous",
    "trustworthy",
    "authoritative",
}

_RECENCY_HINTS = {
    "recent",
    "latest",
    "newest",
    "current",
    "up-to-date",
}

_BENCHMARK_HINTS = {"benchmark", "benchmarks"}
_SURVEY_HINTS = {"survey", "surveys"}
_WHITEPAPER_HINTS = {"whitepaper", "white paper", "whitepapers"}
_OUTLOOK_HINTS = {"outlook", "forecast", "projection"}

_RANKING_HINTS = {
    "best",
    "top",
    "rank",
    "ranking",
    "most credible",
    "highest quality",
}


def _infer_preferred_report_types(query: str) -> list[str]:
    """Infer which report formats the user appears to want."""
    preferred: list[str] = ["report"]

    if any(hint in query for hint in _BENCHMARK_HINTS):
        preferred.append("benchmark")
    if any(hint in query for hint in _SURVEY_HINTS):
        preferred.append("survey")
    if any(hint in query for hint in _WHITEPAPER_HINTS):
        preferred.append("whitepaper")
    if any(hint in query for hint in _OUTLOOK_HINTS):
        preferred.append("outlook")

    seen: set[str] = set()
    ordered: list[str] = []
    for item in preferred:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _wants_quality_priority(query: str) -> bool:
    """Return True when the query emphasizes credibility or analytical quality."""
    return any(hint in query for hint in _QUALITY_HINTS | _RANKING_HINTS)


def _wants_recency_priority(query: str, year_constraint: int | None) -> bool:
    """Return True when freshness appears important."""
    return year_constraint is not None or any(hint in query for hint in _RECENCY_HINTS)


def _infer_task_type(query: str) -> str:
    """Infer the high-level task requested by the user."""
    if any(hint in query for hint in _RANKING_HINTS):
        return "report_ranking"
    return "report_discovery"


def _build_steps(
    quality_priority: bool,
    recency_priority: bool,
    preferred_report_types: list[str],
) -> list[str]:
    """Build an execution plan that stays close to the current system flow."""
    steps = [
        "search",
        "filter_reports",
    ]

    if quality_priority or len(preferred_report_types) > 1:
        steps.extend(["fetch_top_docs", "parse_docs"])

    steps.extend([
        "extract_signals",
        "score",
    ])

    if recency_priority:
        steps.append("apply_recency_preference")

    steps.extend([
        "rank",
        "explain",
    ])
    return steps


def make_plan(user_query: str) -> dict[str, Any]:
    """Convert a raw user request into a structured plan for the report agent.

    - extract a year if one is present
    - infer whether the user cares about quality/credibility
    - infer preferred report formats such as benchmark or survey
    - choose a suggested execution sequence for downstream tools
    """
    normalized_query = _normalize_query_text(user_query)
    years = _extract_years(normalized_query)
    year_constraint = years[0] if years else None
    quality_priority = _wants_quality_priority(normalized_query)
    recency_priority = _wants_recency_priority(normalized_query, year_constraint)
    preferred_report_types = _infer_preferred_report_types(normalized_query)

    plan: dict[str, Any] = {
        "task_type": _infer_task_type(normalized_query),
        "topic": _base_topic(normalized_query) or "industry report",
        "year_constraint": year_constraint,
        "quality_priority": quality_priority,
        "recency_priority": recency_priority,
        "preferred_report_types": preferred_report_types,
        "needs_ranking": any(hint in normalized_query for hint in _RANKING_HINTS),
        "steps": _build_steps(
            quality_priority=quality_priority,
            recency_priority=recency_priority,
            preferred_report_types=preferred_report_types,
        ),
    }
    return plan


if __name__ == "__main__":
    examples = [
        "Find the most credible enterprise AI adoption benchmark for 2025",
        "Show me recent renewable energy market surveys",
        "Best semiconductor industry benchmark 2024",
        "cybersecurity threat report",
    ]

    for example in examples:
        print(f"\nQuery: {example}")
        print(make_plan(example))
