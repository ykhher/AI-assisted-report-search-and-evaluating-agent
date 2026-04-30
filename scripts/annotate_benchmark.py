"""Annotate curated benchmark documents with local Qwen quality labels.

This script rewrites data/curated_benchmark/quality_annotations.csv using the
local Qwen model as the judge for every annotation metric. The only derived
values are quality_score and ideal_final_score, which are computed from the
LLM-provided metric scores so they remain compatible with the existing tuning
and evaluation code.

Run:
    $env:USE_LOCAL_QWEN='1'; python scripts/annotate_benchmark.py
    $env:USE_LOCAL_QWEN='1'; python scripts/annotate_benchmark.py --dry-run
    $env:USE_LOCAL_QWEN='1'; python scripts/annotate_benchmark.py --query q003
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

SCORE_FIELDS = [
    "relevance_score",
    "report_validity_score",
    "methodology_score",
    "citation_score",
    "consistency_score",
    "structure_score",
    "data_density",
    "claim_density",
    "authority_score",
]

QUALITY_FIELDS = [
    "doc_id",
    *SCORE_FIELDS[:8],
    "quality_score",
    "authority_score",
    "ideal_final_score",
    "rationale",
]

QUALITY_WEIGHTS = {
    "methodology_score": 0.22,
    "citation_score": 0.22,
    "consistency_score": 0.18,
    "structure_score": 0.14,
    "data_density": 0.14,
    "claim_density": 0.10,
}

FINAL_WEIGHTS = {
    "relevance_score": 0.35,
    "report_validity_score": 0.20,
    "quality_score": 0.30,
    "authority_score": 0.15,
}

SYSTEM_PROMPT = """\
You are an expert benchmark annotator for industry-report discovery.
Score the candidate document against the user query.

Return only one strict JSON object. All numeric scores must be floats from 0.0
to 1.0. Do not copy any fixed heuristic table. Use the document metadata, title,
snippet, fetched text excerpt, and query intent to judge every metric.

Be conservative. Score only evidence that is visible in the prompt. Do not infer
that methodology, citations, charts, or report sections exist just because the
title sounds like a report. If fetched text is missing, rely on the metadata and
snippet, and lower evidence-dependent metrics when the snippet does not show the
evidence. Treat document_type metadata such as landing_page, news_article, and
archived_pdf as meaningful evidence about report validity and structure.

Metrics:
- relevance_score: topical, industry, geography, and year match with the query.
- report_validity_score: whether this is a credible report-like source rather
  than a landing page, generic web page, news item, encyclopedia page, or ad.
- methodology_score: evidence of research design, survey, benchmark method,
  sample, data collection, analytical framework, or transparent assumptions.
- citation_score: references, footnotes, source notes, named data providers,
  institutional attribution, links to supporting evidence, or bibliography.
- consistency_score: coherent, internally consistent claims and conclusions.
- structure_score: report organization, sections, tables/charts, executive
  summary, findings, methods, appendices, or other report-like structure.
- data_density: density of specific numbers such as percentages, market size,
  investment dollars, CAGR, sample sizes, time ranges, or rankings.
- claim_density: density of analytical/forecasting claims relevant to the query.
- authority_score: credibility and domain authority of the publisher/source.

Calibration:
- A promotional landing page with no visible sources or method should receive
  low report_validity_score, methodology_score, citation_score, structure_score,
  and consistency_score even when it is topically relevant.
- A news article can be relevant and authoritative, but it is usually not a
  benchmark/research report unless the text itself shows report structure.
- A research report from a credible institution may score high, but only if the
  prompt shows evidence of method, citations, data, and report-like structure.

JSON schema:
{
  "relevance_score": 0.0,
  "report_validity_score": 0.0,
  "methodology_score": 0.0,
  "citation_score": 0.0,
  "consistency_score": 0.0,
  "structure_score": 0.0,
  "data_density": 0.0,
  "claim_density": 0.0,
  "authority_score": 0.0,
  "rationale": "one concise sentence explaining the most important scoring reason"
}
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


def _excerpt(text: str, max_chars: int = 2600) -> str:
    text = str(text or "")
    text = re.sub(r"(?is)<script\b.*?</script>", " ", text)
    text = re.sub(r"(?is)<style\b.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    head = text[:1800].rstrip()
    tail = text[-600:].lstrip()
    return f"{head}\n...\n{tail}"


def _coerce_annotation(parsed: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field in SCORE_FIELDS:
        row[field] = round(_clamp01(parsed.get(field)), 3)

    rationale = str(parsed.get("rationale") or "").strip()
    row["rationale"] = re.sub(r"\s+", " ", rationale)[:500]

    quality = sum(row[field] * weight for field, weight in QUALITY_WEIGHTS.items())
    row["quality_score"] = round(quality, 4)

    ideal = sum(row[field] * weight for field, weight in FINAL_WEIGHTS.items())
    row["ideal_final_score"] = round(ideal, 4)
    return row


def _build_prompt(doc: dict[str, str], query: dict[str, str], fetched_text: str) -> str:
    query_text = query.get("query", "")
    target_min = query.get("target_year_min", "")
    target_max = query.get("target_year_max", "")
    return (
        f"Query: {query_text}\n"
        f"Query topic: {query.get('topic', '')}\n"
        f"Industry: {query.get('industry', '')}\n"
        f"Geography: {query.get('geography', '')}\n"
        f"Target year range: {target_min}-{target_max}\n"
        f"Intent: {query.get('intent', '')}\n\n"
        "Document metadata:\n"
        f"doc_id: {doc.get('doc_id', '')}\n"
        f"title: {doc.get('title', '')}\n"
        f"source: {doc.get('source', '')}\n"
        f"source_type metadata: {doc.get('source_type', '')}\n"
        f"url: {doc.get('url', '')}\n"
        f"year: {doc.get('year', '')}\n"
        f"document_type metadata: {doc.get('document_type', '')}\n"
        f"is_pdf: {doc.get('is_pdf', '')}\n"
        f"snippet: {doc.get('snippet', '')}\n\n"
        "Fetched text excerpt:\n"
        f"{_excerpt(fetched_text) or '[no fetched text available]'}\n\n"
        "Score every metric from the evidence above and return JSON only."
    )


def annotate_document(
    doc: dict[str, str],
    query: dict[str, str],
    fetched_text: str,
    retries: int = 2,
) -> dict[str, Any]:
    prompt = _build_prompt(doc, query, fetched_text)
    for attempt in range(1, retries + 1):
        raw = _generate(prompt, max_new_tokens=450, system_prompt=SYSTEM_PROMPT)
        parsed = _parse_json_object(raw)
        if parsed:
            annotation = _coerce_annotation(parsed)
            if annotation["rationale"]:
                return annotation
            annotation["rationale"] = "LLM returned scores without a rationale."
            return annotation
        print(f"  [warn] no parseable JSON for {doc.get('doc_id')} on attempt {attempt}")

    repair_prompt = (
        "The previous response was not valid JSON. Based only on this document, "
        "return the required JSON object now. Do not repeat document text.\n\n"
        f"{prompt}"
    )
    raw = _generate(repair_prompt, max_new_tokens=450, system_prompt=SYSTEM_PROMPT)
    parsed = _parse_json_object(raw)
    if parsed:
        annotation = _coerce_annotation(parsed)
        annotation["rationale"] = annotation["rationale"] or "LLM repaired a non-JSON response."
        return annotation

    raise RuntimeError(f"Qwen did not return parseable JSON for {doc.get('doc_id')}")


def _load_csv(path: Path, key: str | None = None) -> Any:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if key:
        return {row[key]: row for row in rows}
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=QUALITY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run(target_query: str | None = None, dry_run: bool = False) -> None:
    if not _local_qwen_enabled():
        raise SystemExit("Set USE_LOCAL_QWEN=1 before running this script.")

    queries = _load_csv(DATA_DIR / "queries.csv", key="query_id")
    documents = _load_csv(DATA_DIR / "documents.csv")
    fetched_texts = _load_csv(DATA_DIR / "document_texts.csv", key="doc_id")

    rows: list[dict[str, Any]] = []
    for doc in documents:
        if target_query and doc["query_id"] != target_query:
            continue
        query = queries.get(doc["query_id"], {})
        text_row = fetched_texts.get(doc["doc_id"], {})
        print(f"  Annotating {doc['doc_id']} with Qwen")
        annotation = annotate_document(doc, query, text_row.get("fetched_text", ""))
        rows.append({"doc_id": doc["doc_id"], **annotation})

    if dry_run:
        print(f"\nDry run complete: {len(rows)} document(s) annotated.")
        for row in rows[:3]:
            print(json.dumps(row, indent=2))
        return

    if target_query:
        existing = _load_csv(DATA_DIR / "quality_annotations.csv", key="doc_id")
        for row in rows:
            existing[row["doc_id"]] = row
        rows = list(existing.values())

    _write_csv(DATA_DIR / "quality_annotations.csv", rows)
    print(f"\nDone. Wrote {len(rows)} rows to {DATA_DIR / 'quality_annotations.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate benchmark documents with local Qwen.")
    parser.add_argument("--query", metavar="QUERY_ID", help="Annotate only one query, e.g. q003.")
    parser.add_argument("--dry-run", action="store_true", help="Print sample annotations without saving.")
    args = parser.parse_args()
    run(target_query=args.query, dry_run=args.dry_run)
