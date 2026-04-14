"""Score the local dataset using the current agent-style ranking path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow direct execution with `python test/test_dataset_scores.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.classifier.report_classifier import classify_report_type
from source.classifier.source_classifier import classify_source
from source.extractor import extract_signals, source_score
from source.scoring import compute_relevance_score, rank_reports


DATASET_PATH = PROJECT_ROOT / "data" / "dataset.json"


def _is_pdf_url(url: str) -> bool:
    """Return True when a dataset URL looks like a PDF document."""
    lowered = str(url or "").lower()
    return lowered.endswith(".pdf") or "/pdf" in lowered


def build_proxy_text(item: dict[str, Any]) -> str:
    """Create lightweight surrogate text from the dataset metadata.

    The dataset does not include full document text, so this test builds a
    small proxy text block that still looks like the payload shape the agent
    uses during ranking.
    """
    parts = [
        str(item.get("title", "")).strip(),
        str(item.get("source", "")).strip(),
    ]

    if item.get("has_forecast"):
        parts.append("forecast outlook report analysis projections")

    if _is_pdf_url(str(item.get("url", ""))):
        parts.append("pdf report methodology references")

    return ". ".join(part for part in parts if part).strip()


def _build_candidate(item: dict[str, Any], query: str) -> dict[str, Any]:
    """Convert one dataset row into the candidate shape used by rank_reports."""
    url = str(item.get("url", "")).strip()
    title = str(item.get("title", "")).strip()
    source = str(item.get("source", "")).strip()
    text = build_proxy_text(item)
    metadata = {
        "year": item.get("year"),
        "source": source,
        "is_pdf": _is_pdf_url(url),
    }

    signals = extract_signals(text, metadata)
    signals["source_name"] = source
    signals["source"] = source_score(source, text)
    signals["_text"] = text

    source_info = classify_source(url=url, title=title, text=text)
    report_info = classify_report_type(title=title, text=text, metadata=metadata)

    signals["source_class"] = source_info.get("source_class", "unknown")
    signals["authority_prior"] = source_info.get("authority_prior", 0.45)
    signals["report_type"] = report_info.get("report_type", "unknown")
    signals["report_validity_score_classifier"] = report_info.get("report_validity_score", 0.0)

    return {
        "id": item.get("id"),
        "title": title,
        "url": url,
        "source": source,
        "year": item.get("year"),
        "snippet": text,
        "text": text,
        "is_pdf": metadata["is_pdf"],
        "relevance": compute_relevance_score(query, text),
        "signals": signals,
    }


def score_dataset(query: str) -> list[dict[str, Any]]:
    """Load dataset entries, map them to agent candidates, and rank them."""
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    candidates = [_build_candidate(item, query) for item in dataset if isinstance(item, dict)]
    ranked = rank_reports(candidates, top_k=None, query=query)
    candidate_by_url = {str(candidate.get("url", "")).strip(): candidate for candidate in candidates}

    results: list[dict[str, Any]] = []
    for ranked_item in ranked:
        candidate = candidate_by_url.get(str(ranked_item.get("url", "")).strip(), {})
        results.append(
            {
                "title": ranked_item.get("title"),
                "source": candidate.get("source"),
                "year": candidate.get("year"),
                "url": candidate.get("url"),
                "report_type": ranked_item.get("report_type"),
                "source_class": ranked_item.get("source_class"),
                "score": ranked_item.get("score"),
                "RQI": ranked_item.get("RQI"),
                "reason": ranked_item.get("reason"),
            }
        )

    results.sort(key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the dataset scoring harness."""
    parser = argparse.ArgumentParser(description="Score the local dataset with the current ranking logic.")
    parser.add_argument("query", nargs="*", help="Query to score the dataset against.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    query = " ".join(args.query).strip() or "AI market report"
    results = score_dataset(query)

    print(f"\nTop {min(10, len(results))} scored reports for query: {query}\n")
    for row in results[:10]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
