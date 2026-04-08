"""Compute heuristic RQI and final score for every entry in `dataset.json`."""

from __future__ import annotations

import json
import re
from pathlib import Path

from extractor import extract_signals, source_score
from scoring import compute_rqi, final_score, generate_reason

DATASET_PATH = Path(__file__).with_name("dataset.json")


def compute_relevance(query: str, text: str) -> float:
    """Simple keyword-overlap relevance used for dataset-level testing."""
    generic_terms = {
        "pdf", "report", "reports", "2024", "2025", "industry", "analysis",
        "market", "forecast", "outlook", "cagr", "projection", "research",
        "global", "value", "top",
    }
    query_words = {
        word for word in re.findall(r"[a-z0-9]+", query.lower())
        if len(word) > 2 and word not in generic_terms
    }
    if not query_words:
        return 0.0

    text_words = set(re.findall(r"[a-z0-9]+", text.lower()))
    overlap = len(query_words & text_words)
    return round(min(overlap / len(query_words), 1.0), 3)


def build_proxy_text(item: dict) -> str:
    """Create lightweight surrogate text from the dataset metadata."""
    parts = [
        item.get("title", ""),
        item.get("source", ""),
    ]

    if item.get("has_forecast"):
        parts.append("forecast outlook report analysis")

    url = str(item.get("url", ""))
    if url.lower().endswith(".pdf") or "/pdf" in url.lower():
        parts.append("pdf report")

    return ". ".join(part for part in parts if part).strip()


def score_dataset(query: str) -> list[dict]:
    """Load `dataset.json`, compute signals and scores, and return ranked results."""
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    scored = []

    for item in dataset:
        text = build_proxy_text(item)
        metadata = {
            "year": item.get("year"),
            "source": item.get("source", ""),
            "is_pdf": ".pdf" in str(item.get("url", "")).lower() or "/pdf" in str(item.get("url", "")).lower(),
        }

        signals = extract_signals(text, metadata)
        signals["source_name"] = str(metadata["source"])
        signals["source"] = source_score(metadata["source"], text)
        signals["_text"] = text

        relevance = compute_relevance(query, text)
        rqi = compute_rqi(signals, text)
        score = final_score(relevance, rqi)
        reason = generate_reason(signals, relevance=relevance)

        scored.append({
            "id": item.get("id"),
            "title": item.get("title"),
            "source": item.get("source"),
            "year": item.get("year"),
            "RQI": rqi,
            "score": score,
            "reason": reason,
        })

    scored.sort(key=lambda row: row["score"], reverse=True)
    return scored


if __name__ == "__main__":
    query = input("Enter dataset scoring query: ").strip() or "AI market report"
    results = score_dataset(query)

    print(f"\nTop {min(10, len(results))} scored reports for query: {query}\n")
    for row in results[:10]:
        print(json.dumps(row, ensure_ascii=False))
