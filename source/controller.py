"""Deterministic single-agent controller for report discovery and ranking.

    plan
    -> choose next action
    -> call tool(s)
    -> update state
    -> reflect
    -> replan or stop

Example flow:
    # [controller] step=1 action=search
    # [controller] step=2 action=filter_reports
    # [controller] step=3 action=fetch_top_docs
    # [controller] step=4 action=parse_docs
    # [controller] reflection: Step=reflect, usable_candidates=4, avg_final_score=0.671, ...
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

from source.agent_state import AgentState
from source.extractor import is_report
from source.filter import filter_results
from source.planner import make_plan
from source.reflection import diagnose_failure, should_stop, summarize_progress
from source.runtime.iteration_controller import rewrite_from_failure
from source.scoring import compute_verification_adjusted_final_score
from source.tool_registry import get_tool_registry


CONTROLLER_ACTIONS = [
    "search",
    "filter_reports",
    "fetch_top_docs",
    "parse_docs",
    "classify_candidates",
    "extract_signals",
    "score_candidates",
    "rank_candidates",
    "verify_top_reports",
    "reflect",
]

MAX_REPLANS = 2
DEFAULT_SEARCH_COUNT = 12
DEFAULT_FETCH_LIMIT = 5
DEFAULT_VERIFY_TOP_N = 3
DEFAULT_MAX_STEPS = 30


def _plan_int(plan: dict[str, Any], key: str, default: int) -> int:
    """Read an integer plan option while preserving explicit zero values."""
    value = plan.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """Return items without duplicates while preserving first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return ordered


def _normalize_plan_steps(plan: dict[str, Any]) -> list[str]:
    """Translate planner steps into controller actions.

    The planner stays user-facing and broad, while the controller uses a small
    fixed action vocabulary. This function bridges the two without adding a
    framework layer.
    """
    step_map = {
        "search": "search",
        "filter_reports": "filter_reports",
        "fetch_top_docs": "fetch_top_docs",
        "parse_docs": "parse_docs",
        "extract_signals": "extract_signals",
        "score": "score_candidates",
        "rank": "rank_candidates",
        "explain": "reflect",
        "reflect": "reflect",
    }

    planned_actions: list[str] = []
    for step in plan.get("steps", []):
        action = step_map.get(str(step).strip())
        if action and action not in planned_actions:
            planned_actions.append(action)

    # Classification is useful for scoring but is not yet exposed by planner.
    if "classify_candidates" not in planned_actions:
        insert_at = planned_actions.index("extract_signals") if "extract_signals" in planned_actions else len(planned_actions)
        planned_actions.insert(insert_at, "classify_candidates")

    if "reflect" not in planned_actions:
        planned_actions.append("reflect")

    verify_top_n = int(plan.get("verify_top_n", DEFAULT_VERIFY_TOP_N) or 0)
    if verify_top_n > 0 and "verify_top_reports" not in planned_actions:
        insert_at = planned_actions.index("reflect") if "reflect" in planned_actions else len(planned_actions)
        planned_actions.insert(insert_at, "verify_top_reports")

    return [action for action in CONTROLLER_ACTIONS if action in planned_actions]


def _current_query(state: AgentState) -> str:
    """Return the active working query for this run."""
    if state.rewritten_queries_tried:
        return state.rewritten_queries_tried[-1]
    return str(state.current_plan.get("planning_query") or state.user_query)


def _query_keywords(query: str) -> list[str]:
    """Extract a few non-trivial query words for keyword filtering."""
    generic_terms = {
        "report", "reports", "pdf", "analysis", "industry", "market", "research",
        "forecast", "outlook", "benchmark", "survey",
    }
    words = [
        word
        for word in re.findall(r"[a-z0-9]+", str(query).lower())
        if len(word) > 2 and word not in generic_terms
    ]
    return words[:6]


def _candidate_payloads(state: AgentState) -> list[dict[str, Any]]:
    """Return active candidate payload dictionaries from state."""
    return state.active_candidate_payloads()


def _select_fetch_targets(state: AgentState, limit: int = DEFAULT_FETCH_LIMIT) -> list[dict[str, Any]]:
    """Choose the next small batch of candidates to fetch more deeply."""
    targets: list[dict[str, Any]] = []
    for payload in _candidate_payloads(state):
        if payload.get("raw_text"):
            continue
        if payload.get("url") in state.failed_urls:
            continue
        targets.append(payload)
        if len(targets) >= limit:
            break
    return targets


def _infer_year(candidate: dict[str, Any]) -> int | None:
    """Infer a year from candidate fields when no explicit year is present."""
    direct_year = candidate.get("year")
    if isinstance(direct_year, int):
        return direct_year

    for field in ("date", "title", "url", "snippet"):
        match = re.search(r"\b(?:19|20)\d{2}\b", str(candidate.get(field, "")))
        if match:
            return int(match.group(0))
    return None


def _topic_relevance(query: str, text: str) -> float:
    """Compute a lightweight query overlap score for filtering and scoring prep."""
    query_terms = set(_query_keywords(query))
    if not query_terms:
        return 0.0

    text_terms = set(re.findall(r"[a-z0-9]+", str(text).lower()))
    overlap = len(query_terms & text_terms) / len(query_terms)
    return round(min(overlap, 1.0), 3)


def _candidate_score(candidate: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Read a score from a ranked candidate or its score_breakdown."""
    try:
        if key in candidate:
            return float(candidate.get(key, default) or default)
        breakdown = candidate.get("score_breakdown", {})
        if isinstance(breakdown, dict):
            return float(breakdown.get(key, default) or default)
    except (TypeError, ValueError):
        return default
    return default


def _average_score(candidates: list[dict[str, Any]], key: str) -> float:
    """Average a candidate score field safely."""
    if not candidates:
        return 0.0
    return sum(_candidate_score(candidate, key) for candidate in candidates) / len(candidates)


def _ranked_result_quality(ranked_results: list[dict[str, Any]]) -> float:
    """Score a ranked result set so the controller can keep the best iteration."""
    if not ranked_results:
        return 0.0

    top_results = ranked_results[:3]
    top_final = max(_candidate_score(item, "final_score", _candidate_score(item, "score")) for item in top_results)
    avg_final = _average_score(top_results, "final_score") or _average_score(top_results, "score")
    avg_quality = _average_score(top_results, "quality_score")
    avg_validity = _average_score(top_results, "report_validity_score")
    avg_relevance = _average_score(top_results, "relevance_score")

    return round(
        0.35 * top_final
        + 0.25 * avg_final
        + 0.20 * avg_quality
        + 0.10 * avg_validity
        + 0.10 * avg_relevance,
        6,
    )


def _rerank_after_verification(ranked_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rerank verified reports using verification-adjusted quality scores."""
    reranked: list[dict[str, Any]] = []
    for index, item in enumerate(ranked_results):
        enriched = dict(item)
        adjusted_breakdown = compute_verification_adjusted_final_score(enriched)
        if enriched.get("verification_adjusted_quality_score") is not None:
            enriched["pre_verification_score"] = enriched.get("score", adjusted_breakdown["final_score"])
            enriched["pre_verification_score_breakdown"] = dict(enriched.get("score_breakdown", {}) or {})
            enriched["score_breakdown"] = adjusted_breakdown
            enriched["score"] = adjusted_breakdown["final_score"]
            enriched["verification_reranked"] = True
        else:
            enriched.setdefault("verification_reranked", False)
        enriched["_previous_rank"] = index + 1
        reranked.append(enriched)

    reranked.sort(
        key=lambda item: (
            float(item.get("score", 0.0) or 0.0),
            -int(item.get("_previous_rank", 0) or 0),
        ),
        reverse=True,
    )
    for index, item in enumerate(reranked, start=1):
        previous_rank = int(item.pop("_previous_rank", index) or index)
        item["rank"] = index
        if item.get("verification_reranked"):
            item["pre_verification_rank"] = previous_rank
    return reranked


def _build_search_query(state: AgentState) -> str:
    """Build the current search query from plan + rewritten attempts."""
    active_query = _current_query(state).strip()
    if not active_query:
        active_query = str(state.current_plan.get("topic") or state.user_query).strip()

    query_parts = [active_query]

    year = state.current_plan.get("year_constraint")
    if year and str(year) not in active_query:
        query_parts.append(str(year))

    preferred_types = state.current_plan.get("preferred_report_types", [])
    if isinstance(preferred_types, list):
        query_parts.extend(str(item) for item in preferred_types[:2])

    source_preferences = state.current_plan.get("preferred_source_classes", [])
    if isinstance(source_preferences, list):
        source_terms_map = {
            "government": "site:.gov",
            "academic": "site:.edu",
            "research_institute": "research institute",
            "consulting": "industry analysis",
        }
        for item in source_preferences[:2]:
            mapped = source_terms_map.get(str(item).strip().lower())
            if mapped:
                query_parts.append(mapped)

    search_hints = state.current_plan.get("search_hints", [])
    if isinstance(search_hints, list):
        query_parts.extend(str(item) for item in search_hints[:3])

    expanded_query = " ".join(part for part in query_parts if str(part).strip()).strip()
    return " ".join(_dedupe_preserve_order(expanded_query.split()))


def revise_plan(state: AgentState, failure_type: str | None) -> dict[str, Any]:
    """Build a revised plan after reflection reports weak progress.

    The first version is intentionally rule-based:
    - rewrite the working query using the existing failure-aware helper
    - adjust plan fields the controller can act on directly
    - change step emphasis when a failure suggests deeper inspection or stricter filtering

    Example:
        # failure_type == "low_authority"
        # -> rewritten query may add academic/research language
        # -> plan prefers trusted source classes
        #
        # failure_type == "weak_quality_signals"
        # -> plan increases fetch depth and keeps fetch/parse before ranking
    """
    current_query = str(state.current_plan.get("planning_query") or state.user_query).strip()
    rewritten_query = rewrite_from_failure(current_query, failure_type, max_additions=3) if failure_type else current_query
    repeated_failure_count = state.failure_history.count(str(failure_type)) if failure_type else 0
    if failure_type == "topic_drift" and repeated_failure_count >= 2:
        rewritten_query = f"{state.user_query} empirical study survey working paper"

    revised_plan = make_plan(rewritten_query)

    revised_plan["planning_query"] = rewritten_query
    revised_plan["replan_reason"] = failure_type
    revised_plan["previous_query"] = current_query

    previous_types = state.current_plan.get("preferred_report_types", [])
    if isinstance(previous_types, list):
        revised_plan["preferred_report_types"] = _dedupe_preserve_order(
            list(revised_plan.get("preferred_report_types", [])) + [str(item) for item in previous_types]
        )

    revised_plan["preferred_source_classes"] = list(state.current_plan.get("preferred_source_classes", []))
    revised_plan["search_hints"] = list(state.current_plan.get("search_hints", []))
    revised_plan["fetch_limit"] = _plan_int(state.current_plan, "fetch_limit", DEFAULT_FETCH_LIMIT)
    revised_plan["verify_top_n"] = _plan_int(state.current_plan, "verify_top_n", DEFAULT_VERIFY_TOP_N)

    if failure_type == "topic_drift":
        revised_plan["task_type"] = "report_ranking"
        revised_plan["quality_priority"] = True
        topic_hints = ["industry report", "analysis", "pdf"]
        if repeated_failure_count >= 2:
            topic_hints = ["empirical study", "survey", "working paper", "research report"]
        revised_plan["search_hints"] = _dedupe_preserve_order(list(revised_plan["search_hints"]) + topic_hints)

    elif failure_type == "not_report_like":
        revised_plan["quality_priority"] = True
        revised_plan["preferred_report_types"] = _dedupe_preserve_order(
            list(revised_plan.get("preferred_report_types", [])) + ["report", "benchmark", "survey", "whitepaper"]
        )
        revised_plan["search_hints"] = _dedupe_preserve_order(
            list(revised_plan["search_hints"]) + ["report", "benchmark", "survey"]
        )
        revised_plan["steps"] = [
            "search",
            "filter_reports",
            "classify_candidates",
            "fetch_top_docs",
            "parse_docs",
            "extract_signals",
            "score",
            "rank",
            "explain",
        ]

    elif failure_type == "low_authority":
        revised_plan["quality_priority"] = True
        revised_plan["preferred_source_classes"] = ["government", "academic", "research_institute"]
        revised_plan["search_hints"] = _dedupe_preserve_order(
            list(revised_plan["search_hints"]) + ["official", "research institute", "government", "academic"]
        )

    elif failure_type == "too_old":
        revised_plan["recency_priority"] = True
        revised_plan["year_constraint"] = max(
            int(revised_plan.get("year_constraint") or 0),
            datetime.now().year - 1,
        )
        revised_plan["search_hints"] = _dedupe_preserve_order(
            list(revised_plan["search_hints"]) + ["latest", "recent"]
        )

    elif failure_type == "weak_quality_signals":
        revised_plan["quality_priority"] = True
        revised_plan["fetch_limit"] = max(int(revised_plan.get("fetch_limit", DEFAULT_FETCH_LIMIT) or DEFAULT_FETCH_LIMIT), 8)
        revised_plan["search_hints"] = _dedupe_preserve_order(
            list(revised_plan["search_hints"]) + ["methodology", "findings", "results"]
        )
        revised_plan["steps"] = [
            "search",
            "filter_reports",
            "fetch_top_docs",
            "parse_docs",
            "classify_candidates",
            "extract_signals",
            "score",
            "rank",
            "explain",
        ]

    elif failure_type == "too_few_results":
        broadened_query = re.sub(r"\b(?:must|only|exact|precise|strict)\b", "", rewritten_query, flags=re.IGNORECASE)
        broadened_query = " ".join(broadened_query.split()) or rewritten_query
        revised_plan = make_plan(broadened_query)
        revised_plan["planning_query"] = broadened_query
        revised_plan["replan_reason"] = failure_type
        revised_plan["previous_query"] = current_query
        revised_plan["search_hints"] = ["market", "industry", "analysis"]
        revised_plan["fetch_limit"] = DEFAULT_FETCH_LIMIT
        revised_plan["verify_top_n"] = _plan_int(state.current_plan, "verify_top_n", DEFAULT_VERIFY_TOP_N)

    elif failure_type == "fetch_failures_dominant":
        revised_plan["fetch_limit"] = 2
        revised_plan["search_hints"] = _dedupe_preserve_order(
            list(revised_plan["search_hints"]) + ["pdf"]
        )
        revised_plan["steps"] = [
            "search",
            "filter_reports",
            "classify_candidates",
            "extract_signals",
            "score",
            "rank",
            "explain",
        ]

    return revised_plan


def choose_next_action(state: AgentState) -> str | None:
    """Choose the next deterministic controller action.

    Rules:
    - start with the normalized planner steps
    - after reflection, either stop or replan
    - after replan, restart from search
    """
    if state.stop_reason:
        return None

    if state.current_step == "plan":
        planned_actions = _normalize_plan_steps(state.current_plan)
        return planned_actions[0] if planned_actions else "search"

    if state.current_step == "reflect":
        if state.stop_reason:
            return None
        if state.failure_history and len(state.rewritten_queries_tried) <= MAX_REPLANS:
            return "replan"
        return None

    if state.current_step == "replan":
        return "search"

    if state.current_step == "rank_candidates":
        verify_top_n = int(state.current_plan.get("verify_top_n", DEFAULT_VERIFY_TOP_N) or 0)
        if verify_top_n > 0:
            return "verify_top_reports"
        return "reflect"

    if state.current_step == "verify_top_reports":
        return "reflect"

    planned_actions = _normalize_plan_steps(state.current_plan)
    if state.current_step in planned_actions:
        current_index = planned_actions.index(state.current_step)
        if current_index + 1 < len(planned_actions):
            return planned_actions[current_index + 1]
        return None

    return None


def apply_action(
    state: AgentState,
    action_name: str,
    tool_registry: dict[str, Any] | None = None,
    runtime_context: dict[str, Any] | None = None,
) -> None:
    """Apply one controller action and update state in place.

    The runtime context stores transient outputs like ranked results. This
    keeps `AgentState` focused on durable execution state rather than every
    derived artifact.
    """
    tools = tool_registry or get_tool_registry()
    context = runtime_context if runtime_context is not None else {}
    state.current_step = action_name

    if action_name == "search":
        search_query = _build_search_query(state)
        results = tools["search"](search_query, count=DEFAULT_SEARCH_COUNT)
        state.rewritten_queries_tried.append(search_query)
        state.record_action("search", detail=f"Ran search for: {search_query}", payload={"result_count": len(results)})

        state.candidates.clear()
        state.filtered_out.clear()
        for result in results:
            state.add_candidate(
                url=str(result.get("url", "")),
                title=str(result.get("title", "")),
                source=str(result.get("source", "")),
                metadata=dict(result),
            )
        context["search_results"] = results
        return

    if action_name == "filter_reports":
        active_query = _current_query(state)
        payloads = _candidate_payloads(state)
        preferred_types = state.current_plan.get("preferred_report_types", [])
        report_keywords = [str(item).lower() for item in preferred_types] if isinstance(preferred_types, list) else []
        keyword_filtered = filter_results(
            payloads,
            min_score=0.0,
            keywords=_dedupe_preserve_order(_query_keywords(active_query) + report_keywords),
        ) or payloads
        kept_urls = {str(item.get("url", "")) for item in keyword_filtered}

        for candidate in list(state.candidates):
            payload = dict(candidate.metadata)
            text = str(payload.get("text") or payload.get("snippet") or "")
            metadata = {
                "is_pdf": bool(payload.get("is_pdf", False) or str(payload.get("url", "")).lower().endswith(".pdf")),
                "source": payload.get("source", candidate.source),
                "year": _infer_year(payload),
            }
            report_type_info = tools["classify_report_type"](
                title=str(payload.get("title", candidate.title)),
                text=text,
                metadata=metadata,
            )
            looks_report_like = (
                candidate.url in kept_urls
                or report_type_info.get("report_validity_score", 0.0) >= 0.45
                or is_report(text, metadata)
            )
            if not looks_report_like:
                state.filter_out_candidate(candidate.url, "filtered out as weak or non-report-like")
            else:
                candidate.metadata.update(report_type_info)

        state.record_action(
            "filter_reports",
            detail="Applied keyword and report-likeness filtering",
            payload={"kept": len(state.candidates), "filtered_out": len(state.filtered_out)},
        )
        return

    if action_name == "fetch_top_docs":
        fetch_limit = int(state.current_plan.get("fetch_limit", DEFAULT_FETCH_LIMIT) or DEFAULT_FETCH_LIMIT)
        fetch_targets = _select_fetch_targets(state, limit=fetch_limit)
        fetched_count = 0

        for target in fetch_targets:
            url = str(target.get("url", "")).strip()
            if not url:
                continue

            fetched = tools["fetch_document"](url)
            candidate = state.get_candidate(url)
            if candidate is None:
                continue

            if fetched.get("status") != "ok":
                state.mark_url_failed(url, str(fetched.get("error", "fetch_failed")))
                continue

            candidate.status = "fetched"
            candidate.metadata.update(fetched)
            if fetched.get("title") and not candidate.title:
                candidate.title = str(fetched["title"])
            fetched_count += 1

        state.record_action(
            "fetch_top_docs",
            detail="Fetched top candidate documents for deeper inspection",
            payload={"attempted": len(fetch_targets), "fetched": fetched_count},
        )
        return

    if action_name == "parse_docs":
        parsed_count = 0
        for candidate in state.candidates:
            payload = candidate.metadata
            raw_text = str(payload.get("raw_text") or "").strip()
            if not raw_text:
                continue
            if payload.get("parsed_document"):
                continue

            parsed = tools["parse_document"](raw_text)
            payload["parsed_document"] = parsed
            stats = parsed.get("stats", {}) if isinstance(parsed, dict) else {}
            sections = parsed.get("sections", {}) if isinstance(parsed, dict) else {}
            if isinstance(stats, dict):
                payload["parsed_word_count"] = int(stats.get("word_count", 0) or 0)
                payload["parsed_has_methodology"] = bool(stats.get("has_methodology", False))
                payload["parsed_has_references"] = bool(stats.get("has_references", False))
                payload["parsed_has_statistics_language"] = bool(stats.get("has_statistics_language", False))
            if isinstance(sections, dict):
                payload["parsed_section_lengths"] = {
                    key: len(str(value).split())
                    for key, value in sections.items()
                    if value
                }
            parsed_count += 1

        state.record_action(
            "parse_docs",
            detail="Parsed fetched document text into sections and stats",
            payload={"parsed": parsed_count},
        )
        return

    if action_name == "classify_candidates":
        classified_count = 0
        for candidate in state.candidates:
            payload = candidate.metadata
            title = str(payload.get("title", candidate.title))
            text = str(payload.get("raw_text") or payload.get("text") or payload.get("snippet") or "")
            metadata = {
                "format": payload.get("content_type", "pdf" if payload.get("is_pdf") else ""),
                "file_type": "pdf" if payload.get("is_pdf") else "",
            }

            source_info = tools["classify_source"](candidate.url, title=title, text=text)
            report_type_info = tools["classify_report_type"](title=title, text=text, metadata=metadata)
            payload.update(source_info)
            payload.update(report_type_info)
            classified_count += 1

        state.record_action(
            "classify_candidates",
            detail="Classified source authority and report type",
            payload={"classified": classified_count},
        )
        return

    if action_name == "extract_signals":
        extracted_count = 0
        active_query = _current_query(state)
        for candidate in state.candidates:
            payload = candidate.metadata
            text = str(payload.get("raw_text") or payload.get("text") or payload.get("snippet") or "").strip()
            if not text:
                continue

            metadata = {
                "is_pdf": bool(payload.get("is_pdf", False) or str(candidate.url).lower().endswith(".pdf")),
                "source": payload.get("source", candidate.source),
                "year": _infer_year(payload),
            }
            signals = tools["extract_signals"](text, metadata)
            signals["_text"] = f"{payload.get('title', candidate.title)} {text}".strip()
            signals["source_name"] = str(payload.get("source", candidate.source))
            signals["source_class"] = payload.get("source_class", "unknown")
            signals["authority_prior"] = payload.get("authority_prior", 0.45)
            signals["report_type"] = payload.get("report_type", "unknown")
            signals["report_validity_score_classifier"] = payload.get("report_validity_score", 0.0)

            payload["signals"] = signals
            payload["relevance"] = _topic_relevance(active_query, signals["_text"])
            payload["year"] = metadata["year"]
            extracted_count += 1

        state.record_action(
            "extract_signals",
            detail="Extracted credibility and report-quality signals",
            payload={"extracted": extracted_count},
        )
        return

    if action_name == "score_candidates":
        payloads = _candidate_payloads(state)
        scored = tools["score_candidates"](payloads, query=_current_query(state))
        score_by_url = {str(item.get("url", "")): item for item in scored}

        for candidate in state.candidates:
            scored_item = score_by_url.get(candidate.url)
            if not scored_item:
                continue
            candidate.score = float(scored_item.get("score", 0.0) or 0.0)
            candidate.status = "scored"
            candidate.metadata.update(scored_item)

        state.record_action(
            "score_candidates",
            detail="Computed score breakdowns for active candidates",
            payload={"scored": len(scored)},
        )
        return

    if action_name == "rank_candidates":
        payloads = _candidate_payloads(state)
        ranked = tools["rank_candidates"](payloads, top_k=10, query=_current_query(state))
        context["ranked_results"] = ranked

        current_quality = _ranked_result_quality(ranked)
        best_quality = float(context.get("best_ranked_quality", 0.0) or 0.0)
        if current_quality > best_quality:
            context["best_ranked_quality"] = current_quality
            context["best_ranked_results"] = list(ranked)
            context["best_query"] = _current_query(state)

        rank_by_url = {str(item.get("url", "")): index + 1 for index, item in enumerate(ranked)}
        ranked_urls = set(rank_by_url)
        for candidate in state.candidates:
            if candidate.url in ranked_urls:
                candidate.status = "ranked"
                candidate.metadata["rank"] = rank_by_url[candidate.url]

        state.record_action(
            "rank_candidates",
            detail="Ranked candidates by final score",
            payload={
                "ranked": len(ranked),
                "current_iteration_quality": current_quality,
                "best_iteration_quality": context.get("best_ranked_quality", current_quality),
            },
        )
        return

    if action_name == "verify_top_reports":
        ranked_results = list(context.get("ranked_results", []))
        verify_top_n = int(state.current_plan.get("verify_top_n", DEFAULT_VERIFY_TOP_N) or 0)
        verified_count = 0

        for index, ranked_item in enumerate(ranked_results[: max(0, verify_top_n)]):
            url = str(ranked_item.get("url", "")).strip()
            candidate = state.get_candidate(url)
            if candidate is None:
                continue

            payload = dict(candidate.metadata)
            text = str(
                payload.get("raw_text")
                or payload.get("text")
                or payload.get("snippet")
                or payload.get("signals", {}).get("_text", "")
            ).strip()
            if not text:
                continue

            claims = tools["extract_key_claims"](text, max_claims=3)
            verified = tools["attach_verification_notes"](
                ranked_item,
                claims,
                text=text,
                signals=payload.get("signals", {}),
            )
            ranked_results[index] = verified
            verified_count += 1

        ranked_results = _rerank_after_verification(ranked_results)
        context["ranked_results"] = ranked_results
        verified_quality = _ranked_result_quality(ranked_results)
        if verified_quality >= float(context.get("best_ranked_quality", 0.0) or 0.0):
            context["best_ranked_quality"] = verified_quality
            context["best_ranked_results"] = list(ranked_results)
            context["best_query"] = _current_query(state)

        state.record_action(
            "verify_top_reports",
            detail="Attached lightweight verification notes to top ranked reports",
            payload={
                "verified": verified_count,
                "verify_top_n": verify_top_n,
                "reranked": True,
                "top_score_after_verification": ranked_results[0].get("score", 0.0) if ranked_results else 0.0,
            },
        )
        return

    if action_name == "reflect":
        ranked = context.get("ranked_results", _candidate_payloads(state))
        summary = summarize_progress(state, ranked)
        failure = diagnose_failure(state, ranked)
        state.add_reflection(summary)
        if failure:
            state.add_failure(failure)

        if should_stop(state, ranked):
            state.mark_stopped("sufficient_quality" if not failure else f"stopped_after_{failure}")
        elif failure and len(state.rewritten_queries_tried) >= MAX_REPLANS + 1:
            state.mark_stopped("max_replans_reached")

        state.record_action(
            "reflect",
            detail=summary,
            payload={
                "failure": failure,
                "current_iteration_quality": _ranked_result_quality(ranked) if isinstance(ranked, list) else 0.0,
                "best_iteration_quality": context.get("best_ranked_quality", 0.0),
                "best_query": context.get("best_query", ""),
            },
        )
        return

    if action_name == "replan":
        if len(state.rewritten_queries_tried) >= MAX_REPLANS + 1:
            state.mark_stopped("max_replans_reached")
            state.record_action("replan", status="skipped", detail="Maximum replans reached")
            return

        failure_type = state.failure_history[-1] if state.failure_history else "too_few_results"
        new_plan = revise_plan(state, failure_type)
        state.current_plan = new_plan
        state.current_step = "replan"
        state.record_action(
            "replan",
            detail=f"Replanned after failure '{failure_type}'",
            payload={
                "rewritten_query": new_plan.get("planning_query", _current_query(state)),
                "search_hints": new_plan.get("search_hints", []),
                "preferred_source_classes": new_plan.get("preferred_source_classes", []),
                "fetch_limit": new_plan.get("fetch_limit", DEFAULT_FETCH_LIMIT),
            },
        )
        return


def run_agent(
    user_query: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    tool_registry: dict[str, Any] | None = None,
    return_state: bool = False,
    verbose: bool = True,
    verify_top_n: int = DEFAULT_VERIFY_TOP_N,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Run the deterministic single-agent loop for one user query."""
    started_at = time.perf_counter()
    tools = tool_registry or get_tool_registry()

    initial_plan = make_plan(user_query)
    initial_plan["planning_query"] = user_query
    initial_plan["verify_top_n"] = max(0, int(verify_top_n))
    state = AgentState(user_query=user_query, current_plan=initial_plan)
    runtime_context: dict[str, Any] = {
        "ranked_results": [],
        "best_ranked_results": [],
        "best_ranked_quality": 0.0,
        "best_query": user_query,
    }

    for step_index in range(1, max_steps + 1):
        action_name = choose_next_action(state)
        if action_name is None:
            break

        if verbose:
            print(f"[controller] step={step_index} action={action_name}")

        apply_action(
            state=state,
            action_name=action_name,
            tool_registry=tools,
            runtime_context=runtime_context,
        )

        if action_name == "reflect" and state.reflection_notes and verbose:
            print(f"[controller] reflection: {state.reflection_notes[-1]}")

        if state.stop_reason:
            break
    else:
        state.mark_stopped("max_steps_reached")

    state.processing_time_ms = round((time.perf_counter() - started_at) * 1000.0, 3)

    ranked_results = runtime_context.get("best_ranked_results") or runtime_context.get("ranked_results", [])
    if return_state:
        return {
            "query": user_query,
            "plan": dict(state.current_plan),
            "ranked_results": ranked_results,
            "stop_reason": state.stop_reason,
            "processing_time_ms": state.processing_time_ms,
            "best_query": runtime_context.get("best_query", user_query),
            "best_ranked_quality": runtime_context.get("best_ranked_quality", 0.0),
            "state": state.to_dict(),
        }
    return ranked_results
