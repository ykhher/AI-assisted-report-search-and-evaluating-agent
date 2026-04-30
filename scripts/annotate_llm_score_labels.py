"""Create local-Qwen labels for validity, quality, and final score only.

Writes:
    data/curated_benchmark/llm_score_labels.csv

The output is intended as an LLM-judged oracle for coefficient tuning. It does
not replace quality_annotations.csv, because that file is also used as the
benchmark feature table by other scripts.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from local_qwen import _generate, _local_qwen_enabled  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "curated_benchmark"
OUTPUT_PATH = DATA_DIR / "llm_score_labels.csv"

FIELDS = [
    "doc_id",
    "query_id",
    "llm_validity_score",
    "llm_quality_score",
    "llm_final_score",
    "rationale",
]

SYSTEM_PROMPT = """\
You are a strict expert judge for an industry-report search benchmark.
Score only three labels for the candidate document against the user's query.

Return exactly one JSON object and nothing else:
{
  "validity_score": 0.0,
  "quality_score": 0.0,
  "final_score": 0.0,
  "rationale": "one concise sentence"
}

All scores must be floats between 0.0 and 1.0.

Definitions:
- validity_score: how valid the candidate is as a report/research/benchmark
  source, independent of whether every quality detail is visible. Penalize
  landing pages, generic articles, ads, outdated pages, and encyclopedias.
- quality_score: analytical quality and credibility of the candidate's content:
  method, evidence, citations/sources, data density, structure, consistency,
  and useful report-like analysis.
- final_score: your overall usefulness score for ranking this candidate for the
  query. Combine relevance, validity, quality, authority, and recency using your
  own judgment. This is the oracle score we will tune formulas against.

Be conservative and evidence-based. Do not invent methodology, citations, or
data that are not visible in the metadata, snippet, or fetched text excerpt.
"""


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _parse_json_object(raw: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", raw or "")
    if not match:
        return {}
    snippet = match.group(0)
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(snippet)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _clean_excerpt(text: str, max_chars: int = 2800) -> str:
    text = str(text or "")
    text = re.sub(r"(?is)<script\b.*?</script>", " ", text)
    text = re.sub(r"(?is)<style\b.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:1900].rstrip()}\n...\n{text[-700:].lstrip()}"


def _load_csv(path: Path, key: str | None = None) -> Any:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if key:
        return {row[key]: row for row in rows}
    return rows


def _build_prompt(doc: dict[str, str], query: dict[str, str], fetched_text: str) -> str:
    return (
        f"Query: {query.get('query', '')}\n"
        f"Topic: {query.get('topic', '')}\n"
        f"Industry: {query.get('industry', '')}\n"
        f"Geography: {query.get('geography', '')}\n"
        f"Target years: {query.get('target_year_min', '')}-{query.get('target_year_max', '')}\n"
        f"Intent: {query.get('intent', '')}\n\n"
        "Candidate document:\n"
        f"doc_id: {doc.get('doc_id', '')}\n"
        f"title: {doc.get('title', '')}\n"
        f"source: {doc.get('source', '')}\n"
        f"source_type: {doc.get('source_type', '')}\n"
        f"url: {doc.get('url', '')}\n"
        f"year: {doc.get('year', '')}\n"
        f"document_type: {doc.get('document_type', '')}\n"
        f"is_pdf: {doc.get('is_pdf', '')}\n"
        f"snippet: {doc.get('snippet', '')}\n\n"
        "Fetched text excerpt:\n"
        f"{_clean_excerpt(fetched_text) or '[no fetched text available]'}\n\n"
        "Return the three-score JSON object now."
    )


def annotate_document(doc: dict[str, str], query: dict[str, str], fetched_text: str) -> dict[str, Any]:
    prompt = _build_prompt(doc, query, fetched_text)
    for attempt in range(1, 4):
        raw = _generate(prompt, max_new_tokens=220, system_prompt=SYSTEM_PROMPT)
        parsed = _parse_json_object(raw)
        if parsed:
            rationale = re.sub(r"\s+", " ", str(parsed.get("rationale") or "")).strip()
            return {
                "doc_id": doc["doc_id"],
                "query_id": doc["query_id"],
                "llm_validity_score": round(_clamp01(parsed.get("validity_score")), 3),
                "llm_quality_score": round(_clamp01(parsed.get("quality_score")), 3),
                "llm_final_score": round(_clamp01(parsed.get("final_score")), 3),
                "rationale": rationale[:500] or "LLM returned scores without a rationale.",
            }
        print(f"  [warn] no JSON for {doc['doc_id']} attempt {attempt}")
    raise RuntimeError(f"Qwen did not return parseable JSON for {doc['doc_id']}")


def run(target_query: str | None = None, dry_run: bool = False) -> None:
    if not _local_qwen_enabled():
        raise SystemExit("Set USE_LOCAL_QWEN=1 before running this script.")

    queries = _load_csv(DATA_DIR / "queries.csv", key="query_id")
    documents = _load_csv(DATA_DIR / "documents.csv")
    fetched = _load_csv(DATA_DIR / "document_texts.csv", key="doc_id")

    rows: list[dict[str, Any]] = []
    for index, doc in enumerate(documents, start=1):
        if target_query and doc["query_id"] != target_query:
            continue
        print(f"  [{index:>3}/{len(documents)}] scoring {doc['doc_id']} with Qwen")
        text = fetched.get(doc["doc_id"], {}).get("fetched_text", "")
        rows.append(annotate_document(doc, queries.get(doc["query_id"], {}), text))

    if dry_run:
        print(f"\nDry run complete: {len(rows)} row(s).")
        for row in rows[:5]:
            print(json.dumps(row, indent=2))
        return

    if target_query and OUTPUT_PATH.exists():
        existing = _load_csv(OUTPUT_PATH, key="doc_id")
        for row in rows:
            existing[row["doc_id"]] = row
        rows = list(existing.values())

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDone. Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Qwen validity/quality/final labels.")
    parser.add_argument("--query", metavar="QUERY_ID", help="Annotate one query only, e.g. q003.")
    parser.add_argument("--dry-run", action="store_true", help="Print rows without saving.")
    args = parser.parse_args()
    run(target_query=args.query, dry_run=args.dry_run)
