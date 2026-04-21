"""Update placeholder URLs in documents.csv using the live SerpApi search.

Usage:
    SERPAPI_API_KEY=<key> python scripts/update_document_urls.py
    python scripts/update_document_urls.py --dry-run   # print changes without writing

For each query in queries.csv this script:
  1. Searches for up to MAX_RESULTS results via the existing search infrastructure.
  2. Classifies each result into one of the five document_type slots.
  3. Replaces the placeholder benchmark.local URL in documents.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from source.search import search_reports  # noqa: E402  (path manipulation above)

DATA_DIR = PROJECT_ROOT / "data" / "curated_benchmark"
QUERIES_CSV = DATA_DIR / "queries.csv"
DOCUMENTS_CSV = DATA_DIR / "documents.csv"

MAX_RESULTS = 10
PLACEHOLDER_HOST = "benchmark.local"

# Document-type slot labels in retrieval_rank order.
SLOT_ORDER = ["research_report", "whitepaper", "landing_page", "news_article", "archived_pdf"]

NEWS_DOMAINS = {
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "cnbc.com",
    "businessinsider.com", "techcrunch.com", "theverge.com", "forbes.com",
    "nytimes.com", "bbc.com", "apnews.com", "axios.com",
}
OLD_YEAR_THRESHOLD = 2021


# ---------------------------------------------------------------------------
# Result classification helpers
# ---------------------------------------------------------------------------

def _year_from_result(result: dict) -> int | None:
    """Best-effort year extraction from date field or URL."""
    import re
    for field in ("date", "year"):
        val = str(result.get(field) or "")
        match = re.search(r"\b(20\d{2})\b", val)
        if match:
            return int(match.group(1))
    url = str(result.get("url") or "")
    match = re.search(r"\b(20\d{2})\b", url)
    if match:
        return int(match.group(1))
    return None


def _is_pdf(result: dict) -> bool:
    url = str(result.get("url") or "").lower()
    return bool(result.get("is_pdf")) or url.endswith(".pdf") or "/pdf" in url


def _is_news(result: dict) -> bool:
    host = urlparse(str(result.get("url") or "")).netloc.lstrip("www.")
    return host in NEWS_DOMAINS


def _is_landing_page(result: dict) -> bool:
    url = str(result.get("url") or "").lower()
    landing_signals = ["/product", "/solutions", "/platform", "/pricing", "/demo", "/get-started"]
    if any(sig in url for sig in landing_signals):
        return True
    snippet = str(result.get("snippet") or "").lower()
    return "request a demo" in snippet or "sign up" in snippet or "free trial" in snippet


def _classify(result: dict) -> str:
    """Assign one of the five document_type labels to a search result."""
    is_pdf = _is_pdf(result)
    is_news = _is_news(result)
    is_landing = _is_landing_page(result)
    year = _year_from_result(result)

    if is_news:
        return "news_article"
    if is_landing and not is_pdf:
        return "landing_page"
    if is_pdf and year and year < OLD_YEAR_THRESHOLD:
        return "archived_pdf"
    if is_pdf:
        return "research_report"  # default best-PDF bucket
    return "whitepaper"


def assign_to_slots(results: list[dict]) -> dict[str, dict | None]:
    """Fill each document_type slot with the first matching result."""
    slots: dict[str, dict | None] = {t: None for t in SLOT_ORDER}

    # First pass: strict classification
    for result in results:
        doc_type = _classify(result)
        if slots[doc_type] is None:
            slots[doc_type] = result

    # Second pass: fill empty slots with leftover results (relaxed)
    used_urls = {r["url"] for r in slots.values() if r}
    leftovers = [r for r in results if r.get("url") not in used_urls]

    for slot in SLOT_ORDER:
        if slots[slot] is None and leftovers:
            slots[slot] = leftovers.pop(0)

    return slots


# ---------------------------------------------------------------------------
# CSV update logic
# ---------------------------------------------------------------------------

def load_queries(path: Path) -> dict[str, str]:
    """Return {query_id: query_text}."""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row["query_id"]: row["query"] for row in reader}


def load_documents(path: Path) -> tuple[list[str], list[dict]]:
    """Return (fieldnames, rows)."""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def save_documents(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_placeholder(url: str) -> bool:
    return PLACEHOLDER_HOST in str(url)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> None:
    queries = load_queries(QUERIES_CSV)
    fieldnames, rows = load_documents(DOCUMENTS_CSV)

    # Group rows by query_id, preserving order.
    query_rows: dict[str, list[dict]] = {}
    for row in rows:
        query_rows.setdefault(row["query_id"], []).append(row)

    changes: list[tuple[str, str, str]] = []  # (doc_id, old_url, new_url)

    for query_id, query_text in queries.items():
        doc_rows = query_rows.get(query_id, [])
        if not doc_rows:
            continue

        # Skip if all URLs already replaced.
        if all(not is_placeholder(r["url"]) for r in doc_rows):
            print(f"[{query_id}] Already updated — skipping.")
            continue

        print(f"[{query_id}] Searching: {query_text!r} …")
        results = search_reports(query_text, count=MAX_RESULTS)

        if not results:
            print(f"[{query_id}] No results returned.")
            continue

        slots = assign_to_slots(results)

        # Match each document row to its slot by document_type.
        for row in doc_rows:
            if not is_placeholder(row["url"]):
                continue
            doc_type = row.get("document_type", "")
            matched = slots.get(doc_type)
            if matched:
                old_url = row["url"]
                row["url"] = matched["url"]
                changes.append((row["doc_id"], old_url, matched["url"]))
                print(f"  [{row['doc_id']}] {doc_type}: {matched['url']}")
            else:
                print(f"  [{row['doc_id']}] {doc_type}: no result found, keeping placeholder.")

    if dry_run:
        print(f"\nDry run — {len(changes)} URL(s) would be updated.")
    else:
        save_documents(DOCUMENTS_CSV, fieldnames, rows)
        print(f"\nSaved. {len(changes)} URL(s) updated in {DOCUMENTS_CSV}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update document URLs via SerpApi search.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing.")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
