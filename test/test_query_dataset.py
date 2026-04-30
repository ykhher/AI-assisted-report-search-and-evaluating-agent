"""Test whether threading planner-extracted topic/year improves retrieval on the benchmark queries.

Runs every query in two modes:
  baseline  — calls search_reports(query) with no pre-extracted values
  planner   — calls search_reports(query, topic=..., year_terms=...) using make_plan() output

Reports Hit@1, Hit@3, MRR, and NDCG@5 for both modes so we can confirm the refactor
does not regress retrieval quality and ideally improves it.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.query.planner import make_plan
from source.search import search_reports


DATA_DIR = PROJECT_ROOT / "data" / "curated_benchmark"
QUERIES_PATH = DATA_DIR / "queries.csv"
QUALITY_PATH = DATA_DIR / "quality_annotations.csv"
TOP_K = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_queries() -> list[dict]:
    with QUERIES_PATH.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_quality_annotations() -> dict[str, float]:
    """Return {doc_id: ideal_final_score}."""
    scores: dict[str, float] = {}
    with QUALITY_PATH.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            doc_id = str(row.get("doc_id") or "").strip()
            try:
                scores[doc_id] = float(row.get("ideal_final_score") or 0.0)
            except (TypeError, ValueError):
                scores[doc_id] = 0.0
    return scores


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _doc_id_from_url(url: str) -> str:
    """Derive the benchmark doc_id from a benchmark.local URL."""
    # URL pattern: https://benchmark.local/<topic>/<doc-slug>
    # doc_id pattern: d001_01, d001_02 ...
    # The curated rows carry doc_id in their source field? No — but the
    # documents.csv encodes doc_id implicitly via url. We reverse-map via the
    # url itself since _curated_rows() doesn't expose doc_id.  Instead, we
    # embed the mapping at load time.
    return ""


def _build_url_to_docid(data_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    docs_path = data_dir / "documents.csv"
    with docs_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            url = str(row.get("url") or "").strip()
            doc_id = str(row.get("doc_id") or "").strip()
            if url and doc_id:
                mapping[url] = doc_id
    return mapping


def _dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def _ndcg(retrieved_doc_ids: list[str], ideal_scores: dict[str, float], k: int) -> float:
    retrieved = retrieved_doc_ids[:k]
    gains = [ideal_scores.get(doc_id, 0.0) for doc_id in retrieved]
    dcg = _dcg(gains)
    ideal = _dcg(sorted(ideal_scores.values(), reverse=True)[:k])
    return round(dcg / ideal, 4) if ideal > 0 else 0.0


def _hit_at_k(retrieved_doc_ids: list[str], relevant_doc_id: str, k: int) -> int:
    return int(relevant_doc_id in retrieved_doc_ids[:k])


def _reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_id: str) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == relevant_doc_id:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Run one query in one mode
# ---------------------------------------------------------------------------

def _run_query(
    query_row: dict,
    url_to_docid: dict[str, str],
    use_plan: bool,
) -> dict[str, Any]:
    query = str(query_row["query"])
    query_id = str(query_row["query_id"])

    if use_plan:
        plan = make_plan(query)
        plan_topic = str(plan.get("topic") or "").strip() or None
        year_constraint = plan.get("year_constraint")
        plan_year_terms = str(year_constraint) if year_constraint else None
        results = search_reports(
            query,
            count=TOP_K,
            use_api=False,
            topic=plan_topic,
            year_terms=plan_year_terms,
        )
    else:
        results = search_reports(query, count=TOP_K, use_api=False)

    retrieved_doc_ids = [
        url_to_docid.get(str(r.get("url") or "").strip(), "")
        for r in results
    ]

    # Gold doc for this query is the research_report (retrieved_rank=1, suffix _01)
    gold_doc_id = f"d{query_id[1:]:>03}_01"

    # Relevant docs for this query (all docs belonging to the query)
    query_num = query_id[1:]  # e.g. "001"
    relevant_doc_ids = {
        doc_id
        for doc_id in url_to_docid.values()
        if doc_id.startswith(f"d{query_num:>03}_")
    }

    return {
        "query_id": query_id,
        "query": query,
        "retrieved_doc_ids": retrieved_doc_ids,
        "gold_doc_id": gold_doc_id,
        "relevant_doc_ids": relevant_doc_ids,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def _compute_metrics(
    run_results: list[dict],
    quality_scores: dict[str, float],
    url_to_docid: dict[str, str],
) -> dict[str, float]:
    hit1_total = 0
    hit3_total = 0
    mrr_total = 0.0
    ndcg_total = 0.0
    n = len(run_results)

    for row in run_results:
        retrieved = row["retrieved_doc_ids"]
        gold = row["gold_doc_id"]
        query_id = row["query_id"]
        query_num = query_id[1:]

        # Per-query ideal scores (only docs belonging to this query)
        query_ideal = {
            doc_id: score
            for doc_id, score in quality_scores.items()
            if doc_id.startswith(f"d{query_num:>03}_")
        }

        hit1_total += _hit_at_k(retrieved, gold, 1)
        hit3_total += _hit_at_k(retrieved, gold, 3)
        mrr_total += _reciprocal_rank(retrieved, gold)
        ndcg_total += _ndcg(retrieved, query_ideal, TOP_K)

    return {
        "hit@1": round(hit1_total / n, 4),
        "hit@3": round(hit3_total / n, 4),
        "MRR":   round(mrr_total / n, 4),
        f"NDCG@{TOP_K}": round(ndcg_total / n, 4),
        "n_queries": n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading benchmark data …")
    queries = _load_queries()
    quality_scores = _load_quality_annotations()
    url_to_docid = _build_url_to_docid(DATA_DIR)

    print(f"Queries: {len(queries)}  |  Annotated docs: {len(quality_scores)}\n")

    baseline_results: list[dict] = []
    planner_results: list[dict] = []

    header = f"{'Query':<12} {'Gold doc':>10}  {'Baseline retrieved':>40}  {'Planner retrieved':>40}"
    print(header)
    print("-" * len(header))

    for row in queries:
        base = _run_query(row, url_to_docid, use_plan=False)
        plan = _run_query(row, url_to_docid, use_plan=True)
        baseline_results.append(base)
        planner_results.append(plan)

        base_ids = " ".join(base["retrieved_doc_ids"][:TOP_K]) or "(none)"
        plan_ids = " ".join(plan["retrieved_doc_ids"][:TOP_K]) or "(none)"
        gold = base["gold_doc_id"]
        base_hit = "Y" if gold in base["retrieved_doc_ids"][:TOP_K] else "N"
        plan_hit = "Y" if gold in plan["retrieved_doc_ids"][:TOP_K] else "N"
        print(f"{row['query_id']:<12} {gold:>10}  [{base_hit}] {base_ids:<38}  [{plan_hit}] {plan_ids}")

    print()
    base_metrics = _compute_metrics(baseline_results, quality_scores, url_to_docid)
    plan_metrics = _compute_metrics(planner_results, quality_scores, url_to_docid)

    col_w = 12
    metric_keys = ["hit@1", "hit@3", "MRR", f"NDCG@{TOP_K}"]
    print(f"{'Metric':<{col_w}} {'Baseline':>{col_w}} {'Planner':>{col_w}} {'Delta':>{col_w}}")
    print("-" * (col_w * 4 + 3))
    for key in metric_keys:
        b = base_metrics[key]
        p = plan_metrics[key]
        delta = round(p - b, 4)
        arrow = "+" if delta > 0 else ("-" if delta < 0 else "=")
        print(f"{key:<{col_w}} {b:>{col_w}.4f} {p:>{col_w}.4f} {arrow}{abs(delta):>{col_w - 1}.4f}")

    print(f"\nQueries evaluated: {base_metrics['n_queries']}")

    if plan_metrics[f"NDCG@{TOP_K}"] >= base_metrics[f"NDCG@{TOP_K}"]:
        print("\nPASS — planner-threaded extraction matches or improves retrieval quality.")
    else:
        print("\nFAIL — planner-threaded extraction regressed retrieval quality.")


if __name__ == "__main__":
    main()
