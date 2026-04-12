"""Run a small benchmark set against the ranking agent."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent import agent_pipeline


@dataclass
class BenchmarkQuery:
    """One benchmark query row loaded from CSV."""

    query_id: str
    query: str


@dataclass
class QueryMetrics:
    """Evaluation metrics for a single query."""

    query_id: str
    query: str
    k: int
    retrieved: int
    precision_at_k: float
    report_validity_rate_at_k: float
    authority_weighted_precision_at_k: float
    average_quality_score_at_k: float
    status: str = "ok"
    error_message: str = ""


def _normalize_text(value: str) -> str:
    """Lowercase and collapse whitespace for robust substring matching."""
    return " ".join(str(value or "").lower().split())


def _to_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float safely with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def call_main_agent(query: str, k: int) -> Any:
    """Call the main pipeline used by benchmark runs."""
    return agent_pipeline(
        query,
        top_k=k,
        output_format="dict",
        export_results=False,
    )


def _extract_results(payload: Any) -> list[dict[str, Any]]:
    """Extract ranked result list from supported agent output shapes."""
    if isinstance(payload, dict):
        results = payload.get("results", [])
        return results if isinstance(results, list) else []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if hasattr(payload, "results") and isinstance(payload.results, list):
        extracted: list[dict[str, Any]] = []
        for item in payload.results:
            if isinstance(item, dict):
                extracted.append(item)
            elif hasattr(item, "to_dict"):
                try:
                    data = item.to_dict()
                    if isinstance(data, dict):
                        extracted.append(data)
                except Exception:
                    continue
        return extracted
    return []


def _in_acceptable_year(year: Any, acceptable_years: list[int]) -> bool:
    """Return True when year matches the acceptable set or when no constraint exists."""
    if not acceptable_years:
        return True
    if isinstance(year, int):
        return year in set(acceptable_years)
    return False


def _domain_in_known_domains(url_or_source: str, known_domains: list[str]) -> bool:
    """Return True if the report domain appears in known relevant domains."""
    if not known_domains:
        return True
    netloc = urlparse(str(url_or_source or "")).netloc or str(url_or_source or "")
    netloc = netloc.lower()
    if not netloc:
        return False
    for domain in known_domains:
        d = str(domain).lower().strip()
        if d and (netloc == d or netloc.endswith("." + d) or d in netloc):
            return True
    return False


def _keyword_match(report: dict[str, Any], expected_keywords: list[str]) -> bool:
    """Return True if at least one expected topic keyword appears in report text fields."""
    if not expected_keywords:
        return True

    text_fields = [
        str(report.get("title", "")),
        str(report.get("reasoning", "")),
        str(report.get("url", "")),
        str(report.get("source", "")),
        str(report.get("report_type", "")),
    ]
    haystack = _normalize_text(" ".join(text_fields))

    for kw in expected_keywords:
        normalized = _normalize_text(kw)
        if normalized and normalized in haystack:
            return True
    return False


def _is_useful_report(report: dict[str, Any], label: dict[str, Any]) -> bool:
    """Return True when a report matches topic and at least one reliability hint."""
    expected_keywords = [str(x) for x in label.get("expected_topic_keywords", [])]
    acceptable_years = [int(x) for x in label.get("acceptable_years", []) if isinstance(x, int)]
    expected_source_classes = {str(x).lower() for x in label.get("expected_source_classes", [])}
    known_domains = [str(x).lower() for x in label.get("known_relevant_domains", [])]

    keyword_ok = _keyword_match(report, expected_keywords)
    if not keyword_ok:
        return False

    year_ok = _in_acceptable_year(report.get("year"), acceptable_years)

    source_class = str(report.get("source_class", "")).lower().strip()
    source_ok = (not expected_source_classes) or (source_class in expected_source_classes)

    url_or_source = str(report.get("url") or report.get("source") or "")
    domain_ok = _domain_in_known_domains(url_or_source, known_domains)

    return year_ok or source_ok or domain_ok


def load_benchmark_queries(csv_path: Path) -> list[BenchmarkQuery]:
    """Load benchmark queries from CSV file."""
    queries: list[BenchmarkQuery] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = str(row.get("query_id", "")).strip()
            query = str(row.get("query", "")).strip()
            if not query_id or not query:
                continue
            queries.append(BenchmarkQuery(query_id=query_id, query=query))
    return queries


def load_benchmark_labels(json_path: Path) -> dict[str, dict[str, Any]]:
    """Load benchmark labels from JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("benchmark_labels.json must contain a top-level object")
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def evaluate_query(query_obj: BenchmarkQuery, label: dict[str, Any], k: int) -> QueryMetrics:
    """Run the agent for one query and compute top-k metrics."""
    payload = call_main_agent(query_obj.query, k=k)
    results = _extract_results(payload)[:k]

    if not results:
        return QueryMetrics(
            query_id=query_obj.query_id,
            query=query_obj.query,
            k=k,
            retrieved=0,
            precision_at_k=0.0,
            report_validity_rate_at_k=0.0,
            authority_weighted_precision_at_k=0.0,
            average_quality_score_at_k=0.0,
            status="empty",
            error_message="",
        )

    useful_flags: list[int] = [1 if _is_useful_report(r, label) else 0 for r in results]
    useful_count = sum(useful_flags)

    validity_values: list[float] = []
    quality_values: list[float] = []
    authority_weighted_hits: list[float] = []

    for idx, report in enumerate(results):
        validity = _to_float(report.get("report_validity_score"))
        if validity <= 0.0:
            breakdown = report.get("score_breakdown")
            if isinstance(breakdown, dict):
                validity = _to_float(breakdown.get("report_validity_score"))
        validity_values.append(validity)

        quality = _to_float(report.get("quality_score"))
        if quality <= 0.0:
            breakdown = report.get("score_breakdown")
            if isinstance(breakdown, dict):
                quality = _to_float(breakdown.get("quality_score"))
        quality_values.append(quality)

        authority = _to_float(report.get("authority_score"))
        if authority <= 0.0:
            breakdown = report.get("score_breakdown")
            if isinstance(breakdown, dict):
                authority = _to_float(breakdown.get("authority_score"))
        authority_weighted_hits.append(authority * useful_flags[idx])

    denom = float(k)
    precision_at_k = useful_count / denom
    report_validity_rate_at_k = sum(1 for v in validity_values if v >= 0.5) / denom
    authority_weighted_precision_at_k = sum(authority_weighted_hits) / denom
    average_quality_score_at_k = (sum(quality_values) / denom) if quality_values else 0.0

    return QueryMetrics(
        query_id=query_obj.query_id,
        query=query_obj.query,
        k=k,
        retrieved=len(results),
        precision_at_k=round(precision_at_k, 4),
        report_validity_rate_at_k=round(report_validity_rate_at_k, 4),
        authority_weighted_precision_at_k=round(authority_weighted_precision_at_k, 4),
        average_quality_score_at_k=round(average_quality_score_at_k, 4),
        status="ok",
        error_message="",
    )


def build_failed_query_metrics(query_obj: BenchmarkQuery, k: int, error: Exception) -> QueryMetrics:
    """Build a zero-metric row when one query run fails."""
    return QueryMetrics(
        query_id=query_obj.query_id,
        query=query_obj.query,
        k=k,
        retrieved=0,
        precision_at_k=0.0,
        report_validity_rate_at_k=0.0,
        authority_weighted_precision_at_k=0.0,
        average_quality_score_at_k=0.0,
        status="failed",
        error_message=f"{type(error).__name__}: {error}",
    )


def run_benchmark(
    queries_csv: Path,
    labels_json: Path,
    k: int,
) -> dict[str, Any]:
    """Run all benchmark queries and return per-query and aggregate metrics."""
    queries = load_benchmark_queries(queries_csv)
    labels = load_benchmark_labels(labels_json)

    per_query: list[QueryMetrics] = []

    for q in queries:
        label = labels.get(q.query_id, {})
        try:
            metrics = evaluate_query(q, label, k=k)
        except Exception as exc:
            metrics = build_failed_query_metrics(q, k=k, error=exc)
        per_query.append(metrics)

    if not per_query:
        raise ValueError("No benchmark queries loaded")

    n = float(len(per_query))
    failed_count = sum(1 for m in per_query if m.status == "failed")
    empty_count = sum(1 for m in per_query if m.status == "empty")
    aggregate = {
        "query_count": int(n),
        "k": k,
        "failed_query_count": failed_count,
        "empty_query_count": empty_count,
        "mean_precision_at_k": round(sum(m.precision_at_k for m in per_query) / n, 4),
        "mean_report_validity_rate_at_k": round(sum(m.report_validity_rate_at_k for m in per_query) / n, 4),
        "mean_authority_weighted_precision_at_k": round(
            sum(m.authority_weighted_precision_at_k for m in per_query) / n,
            4,
        ),
        "mean_average_quality_score_at_k": round(sum(m.average_quality_score_at_k for m in per_query) / n, 4),
    }

    return {
        "aggregate": aggregate,
        "per_query": [m.__dict__ for m in per_query],
    }


def print_report(report: dict[str, Any]) -> None:
    """Print benchmark metrics in a compact human-readable format."""
    aggregate = report.get("aggregate", {})
    per_query = report.get("per_query", [])

    print("=" * 72)
    print("LIGHTWEIGHT AGENT BENCHMARK")
    print("=" * 72)
    print("Aggregate Metrics")
    print("-" * 72)
    print(f"Queries: {aggregate.get('query_count', 0)} | k: {aggregate.get('k', 0)}")
    print(f"Failed Queries: {aggregate.get('failed_query_count', 0)} | Empty Queries: {aggregate.get('empty_query_count', 0)}")
    print(f"Mean Precision@k: {aggregate.get('mean_precision_at_k', 0):.4f}")
    print(f"Mean Report Validity Rate@k: {aggregate.get('mean_report_validity_rate_at_k', 0):.4f}")
    print(f"Mean Authority-Weighted Precision@k: {aggregate.get('mean_authority_weighted_precision_at_k', 0):.4f}")
    print(f"Mean Avg Quality Score@k: {aggregate.get('mean_average_quality_score_at_k', 0):.4f}")

    print("\nPer-Query Metrics")
    print("-" * 72)
    for item in per_query:
        qid = item.get("query_id", "")
        status = item.get("status", "ok")
        p = item.get("precision_at_k", 0.0)
        vr = item.get("report_validity_rate_at_k", 0.0)
        awp = item.get("authority_weighted_precision_at_k", 0.0)
        q = item.get("average_quality_score_at_k", 0.0)
        print(f"{qid}: status={status} | P@k={p:.4f} | Valid@k={vr:.4f} | AWP@k={awp:.4f} | Quality@k={q:.4f}")
        error_message = str(item.get("error_message", "")).strip()
        if error_message:
            print(f"  error: {error_message}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run lightweight benchmark evaluation for the ranking agent.")
    parser.add_argument("--queries", type=Path, default=Path("benchmark_queries.csv"), help="Path to benchmark queries CSV")
    parser.add_argument("--labels", type=Path, default=Path("benchmark_labels.json"), help="Path to benchmark labels JSON")
    parser.add_argument("--k", type=int, default=5, help="Top-k cutoff for metrics")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark_results.json"), help="Output JSON report path")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    if args.k <= 0:
        raise ValueError("--k must be >= 1")
    if not args.queries.exists():
        raise FileNotFoundError(f"Queries file not found: {args.queries}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Labels file not found: {args.labels}")

    report = run_benchmark(args.queries, args.labels, k=args.k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_report(report)
    print(f"\nSaved benchmark report to: {args.output}")


if __name__ == "__main__":
    main()
