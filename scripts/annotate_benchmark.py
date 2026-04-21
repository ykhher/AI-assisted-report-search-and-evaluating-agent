"""Annotate curated_benchmark documents with Qwen-generated DEER quality labels.

Reads  : data/curated_benchmark/documents.csv
         data/curated_benchmark/queries.csv
Writes : data/curated_benchmark/quality_annotations.csv
         data/curated_benchmark/retrieval_annotations.csv

Run:
    USE_LOCAL_QWEN=1 python scripts/annotate_benchmark.py
    USE_LOCAL_QWEN=1 python scripts/annotate_benchmark.py --dry-run   # print first 3 rows
    USE_LOCAL_QWEN=1 python scripts/annotate_benchmark.py --query q003 # single query
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from local_qwen import _generate, _local_qwen_enabled  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "curated_benchmark"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a strict document quality annotator for a retrieval benchmark dataset.
Your job is to evaluate search result documents and assign DEER-framework labels.

DEER stands for:
  D = Documented methodology (is there a described research method, survey, or benchmark process?)
  E = Evidence depth (are there data points, statistics, references, or citations?)
  E = Editorial transparency (are sources, limitations, and scope clearly stated?)
  R = Recency (is the content recent relative to the query's target year range?)

Label scale (integers only):
  3 = Strong / clearly present
  2 = Partial / somewhat present
  1 = Weak / barely present
  0 = Absent / not present

Return ONLY a single JSON object with these exact keys:
{
  "report_validity_label":     <0-3>,
  "deer_method_label":         <0-3>,
  "deer_evidence_label":       <0-3>,
  "deer_transparency_label":   <0-3>,
  "deer_recency_label":        <0-3>,
  "authority_label":           <0-3>,
  "verification_support_label":<0-3>,
  "overall_quality_label":     <0-3>,
  "relevance_label":           <0-3>,
  "result_class":              "<good_report|mediocre_report|weak_page|false_positive|stale_report>",
  "ranking_preference":        <1-5>,
  "rationale":                 "<one concise sentence>"
}

Scoring rules:
- report_validity_label : 3 if it is clearly a structured research/industry report; 0 for a landing page or news snippet.
- deer_method_label      : based on document_type and snippet signals (research_report → likely 2-3; landing_page → 0).
- deer_evidence_label    : presence of statistics, charts, survey data in the snippet.
- deer_transparency_label: whether sources, methodology, and scope are visible.
- deer_recency_label     : 3 if year ≥ query_year_min; 0 if year < (query_year_min - 4).
- authority_label        : credibility of the source_type (intergovernmental/consulting/research_provider → 3; unknown/vendor → 0-1).
- verification_support_label: 3 if PDF with citations; 0 if non-PDF landing page.
- overall_quality_label  : holistic average, rounding to the nearest integer.
- relevance_label        : 3 if document directly matches the query topic; 1 if tangential.
- result_class           : one of the five fixed classes listed above.
- ranking_preference     : 1 = most preferred result for this query; 5 = least preferred.

Do not include any text outside the JSON object.\
"""

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_INT_RANGE = {"report_validity_label", "deer_method_label", "deer_evidence_label",
              "deer_transparency_label", "deer_recency_label", "authority_label",
              "verification_support_label", "overall_quality_label",
              "relevance_label", "ranking_preference"}

_RESULT_CLASSES = {"good_report", "mediocre_report", "weak_page", "false_positive", "stale_report"}


def _clamp(value: object, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(float(str(value)))))
    except (TypeError, ValueError):
        return lo


def _parse_annotation(raw: str) -> dict:
    """Extract a JSON object from Qwen output, tolerating minor formatting issues."""
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {}
    snippet = match.group(0)
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {}


def _coerce(parsed: dict, query_year_min: int) -> dict:
    """Validate and clamp all fields to their expected types/ranges."""
    out: dict = {}
    for key in _INT_RANGE:
        hi = 5 if key == "ranking_preference" else 3
        out[key] = _clamp(parsed.get(key, 0), 0, hi)

    raw_class = str(parsed.get("result_class", "")).strip().lower()
    out["result_class"] = raw_class if raw_class in _RESULT_CLASSES else "false_positive"
    out["rationale"] = str(parsed.get("rationale", "")).strip() or "No rationale provided."
    return out


# ---------------------------------------------------------------------------
# Annotation prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(doc: dict, query: dict) -> str:
    return (
        f"Query: {query['query']}\n"
        f"Query topic: {query['topic']}\n"
        f"Query target years: {query['target_year_min']}–{query['target_year_max']}\n\n"
        f"Document:\n"
        f"  doc_id          : {doc['doc_id']}\n"
        f"  retrieved_rank  : {doc['retrieved_rank']}\n"
        f"  title           : {doc['title']}\n"
        f"  source          : {doc['source']}\n"
        f"  source_type     : {doc['source_type']}\n"
        f"  year            : {doc['year']}\n"
        f"  document_type   : {doc['document_type']}\n"
        f"  is_pdf          : {doc['is_pdf']}\n"
        f"  url             : {doc['url']}\n"
        f"  snippet         : {doc['snippet']}\n\n"
        "Return the JSON annotation object now."
    )


# ---------------------------------------------------------------------------
# Fallback heuristic (used when Qwen is unavailable)
# ---------------------------------------------------------------------------

_TYPE_PROFILES: dict[str, dict] = {
    "research_report": dict(
        report_validity_label=3, deer_method_label=3, deer_evidence_label=3,
        deer_transparency_label=3, deer_recency_label=3, authority_label=3,
        verification_support_label=3, overall_quality_label=3,
        relevance_label=3, result_class="good_report", ranking_preference=1,
    ),
    "whitepaper": dict(
        report_validity_label=2, deer_method_label=1, deer_evidence_label=2,
        deer_transparency_label=1, deer_recency_label=2, authority_label=1,
        verification_support_label=2, overall_quality_label=2,
        relevance_label=2, result_class="mediocre_report", ranking_preference=2,
    ),
    "landing_page": dict(
        report_validity_label=0, deer_method_label=0, deer_evidence_label=0,
        deer_transparency_label=0, deer_recency_label=3, authority_label=1,
        verification_support_label=0, overall_quality_label=0,
        relevance_label=1, result_class="weak_page", ranking_preference=4,
    ),
    "news_article": dict(
        report_validity_label=0, deer_method_label=0, deer_evidence_label=1,
        deer_transparency_label=0, deer_recency_label=3, authority_label=1,
        verification_support_label=1, overall_quality_label=1,
        relevance_label=1, result_class="false_positive", ranking_preference=5,
    ),
    "archived_pdf": dict(
        report_validity_label=2, deer_method_label=1, deer_evidence_label=1,
        deer_transparency_label=1, deer_recency_label=0, authority_label=1,
        verification_support_label=1, overall_quality_label=1,
        relevance_label=1, result_class="stale_report", ranking_preference=3,
    ),
}

_RATIONALES: dict[str, str] = {
    "research_report": "Strong DEER signals: method, evidence, transparency, and recency all present.",
    "whitepaper":      "Some evidence present, but method and transparency are limited.",
    "landing_page":    "No visible method, source support, or verification-friendly evidence.",
    "news_article":    "Recent but not methodical; evidence is mostly quotes rather than report data.",
    "archived_pdf":    "Some structure exists, but weak recency makes it poor for current verification.",
}


def _heuristic_annotation(doc: dict) -> dict:
    doc_type = doc.get("document_type", "news_article")
    profile = _TYPE_PROFILES.get(doc_type, _TYPE_PROFILES["news_article"])
    rationale = _RATIONALES.get(doc_type, "No rationale.")
    return {**profile, "rationale": rationale}


# ---------------------------------------------------------------------------
# Core annotation loop
# ---------------------------------------------------------------------------

# Fields the small model handles reliably vs. those that need type-based overrides.
_QWEN_QUALITY_FIELDS = {
    "report_validity_label", "deer_evidence_label", "deer_transparency_label",
    "deer_recency_label", "authority_label", "verification_support_label",
    "overall_quality_label", "relevance_label", "rationale",
}
# result_class, ranking_preference, and deer_method_label are structurally determined
# by document_type; the 0.5B model consistently mis-classifies these.
_HEURISTIC_OVERRIDE_FIELDS = {"result_class", "ranking_preference", "deer_method_label"}


def _merge(qwen: dict, doc: dict) -> dict:
    """Keep Qwen quality scores; apply heuristic overrides for structural fields."""
    profile = _TYPE_PROFILES.get(doc.get("document_type", "news_article"), _TYPE_PROFILES["news_article"])
    merged = {**qwen}
    for field in _HEURISTIC_OVERRIDE_FIELDS:
        merged[field] = profile[field]
    if not merged.get("rationale") or merged["rationale"] == "No rationale provided.":
        merged["rationale"] = _RATIONALES.get(doc.get("document_type", "news_article"), "No rationale.")
    return merged


def annotate_document(doc: dict, query: dict, use_qwen: bool) -> dict:
    """Return a flat annotation dict for one document."""
    if use_qwen:
        prompt = _build_prompt(doc, query)
        raw = _generate(prompt, max_new_tokens=300, system_prompt=SYSTEM_PROMPT)
        parsed = _parse_annotation(raw)
        if parsed:
            try:
                year_min = int(query.get("target_year_min", 2020))
            except ValueError:
                year_min = 2020
            coerced = _coerce(parsed, year_min)
            return _merge(coerced, doc)
        print(f"  [warn] Qwen returned no parseable JSON for {doc['doc_id']}, using heuristic.")

    return _heuristic_annotation(doc)


def run(target_query: str | None = None, dry_run: bool = False) -> None:
    use_qwen = _local_qwen_enabled()
    if not use_qwen:
        print(
            "[annotate] USE_LOCAL_QWEN not set — running in heuristic-only mode.\n"
            "           Set USE_LOCAL_QWEN=1 to use the local Qwen model."
        )

    # Load inputs
    with (DATA_DIR / "queries.csv").open(newline="", encoding="utf-8") as fh:
        queries = {row["query_id"]: row for row in csv.DictReader(fh)}

    with (DATA_DIR / "documents.csv").open(newline="", encoding="utf-8") as fh:
        documents = list(csv.DictReader(fh))

    quality_rows: list[dict] = []
    retrieval_rows: list[dict] = []
    processed = 0

    for doc in documents:
        qid = doc["query_id"]
        if target_query and qid != target_query:
            continue

        query = queries.get(qid, {})
        print(f"  Annotating {doc['doc_id']} ({doc['document_type']}) …")

        ann = annotate_document(doc, query, use_qwen)

        quality_rows.append({
            "doc_id":                    doc["doc_id"],
            "report_validity_label":     ann["report_validity_label"],
            "deer_method_label":         ann["deer_method_label"],
            "deer_evidence_label":       ann["deer_evidence_label"],
            "deer_transparency_label":   ann["deer_transparency_label"],
            "deer_recency_label":        ann["deer_recency_label"],
            "authority_label":           ann["authority_label"],
            "verification_support_label":ann["verification_support_label"],
            "overall_quality_label":     ann["overall_quality_label"],
            "rationale":                 ann["rationale"],
        })
        retrieval_rows.append({
            "query_id":          qid,
            "doc_id":            doc["doc_id"],
            "relevance_label":   ann["relevance_label"],
            "result_class":      ann["result_class"],
            "ranking_preference":ann["ranking_preference"],
            "rationale":         ann["rationale"],
        })
        processed += 1

    if dry_run:
        print(f"\n--- Dry run: {processed} document(s) annotated (not saved) ---")
        for r in quality_rows[:3]:
            print(json.dumps(r, indent=2))
        return

    # Write quality_annotations.csv
    q_fields = ["doc_id", "report_validity_label", "deer_method_label", "deer_evidence_label",
                "deer_transparency_label", "deer_recency_label", "authority_label",
                "verification_support_label", "overall_quality_label", "rationale"]

    # If updating only one query, merge with existing rows.
    if target_query:
        existing_quality = _load_existing(DATA_DIR / "quality_annotations.csv", "doc_id")
        existing_retrieval = _load_existing(DATA_DIR / "retrieval_annotations.csv", "doc_id")
        for r in quality_rows:
            existing_quality[r["doc_id"]] = r
        for r in retrieval_rows:
            existing_retrieval[r["doc_id"]] = r
        quality_rows = list(existing_quality.values())
        retrieval_rows = list(existing_retrieval.values())

    _write_csv(DATA_DIR / "quality_annotations.csv", q_fields, quality_rows)

    r_fields = ["query_id", "doc_id", "relevance_label", "result_class", "ranking_preference", "rationale"]
    _write_csv(DATA_DIR / "retrieval_annotations.csv", r_fields, retrieval_rows)

    print(f"\nDone. {processed} document(s) annotated.")
    print(f"  quality_annotations.csv   → {DATA_DIR / 'quality_annotations.csv'}")
    print(f"  retrieval_annotations.csv → {DATA_DIR / 'retrieval_annotations.csv'}")


def _load_existing(path: Path, key: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row[key]: row for row in csv.DictReader(fh)}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate benchmark documents with Qwen.")
    parser.add_argument("--query", metavar="QUERY_ID", help="Annotate only one query (e.g. q003).")
    parser.add_argument("--dry-run", action="store_true", help="Print first 3 results without saving.")
    args = parser.parse_args()
    run(target_query=args.query, dry_run=args.dry_run)
