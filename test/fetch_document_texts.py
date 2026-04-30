"""Fetch real document text for each curated benchmark entry.

For each of the 100 benchmark documents, this script:
1. Constructs a targeted DuckDuckGo search query (topic + document type).
2. Tries the top candidate URLs until one yields >= MIN_WORDS words.
3. Writes results to data/curated_benchmark/document_texts.csv.

The output CSV is used by tune_from_llm_labels.py in place of the short
synthetic snippets in documents.csv.

Usage:
    python test/fetch_document_texts.py [--resume]

Options:
    --resume    Skip doc_ids already present in document_texts.csv.
"""

from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.fetching.document_fetcher import fetch_document

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS  # legacy fallback
    except ImportError:
        print("ERROR: ddgs not installed. Run: pip install ddgs")
        sys.exit(1)

DATA_DIR = PROJECT_ROOT / "data" / "curated_benchmark"
OUTPUT_PATH = DATA_DIR / "document_texts.csv"
OUTPUT_FIELDS = ["doc_id", "query_id", "document_type", "fetched_url", "word_count", "status", "fetched_text"]

MIN_WORDS = 150        # minimum to count as usable
MAX_WORDS = 2000       # truncate text at this many words
CANDIDATES_PER_DOC = 5 # how many DDG results to try before giving up
FETCH_TIMEOUT = 12     # seconds per HTTP request
DDG_PAUSE = 2.5        # seconds between DDG API calls (rate-limit safety)

# How to refine DDG query by document type
_TYPE_SUFFIX: dict[str, str] = {
    "research_report": "industry research report analysis findings 2024 2025",
    "whitepaper":      "whitepaper industry report",
    "landing_page":    "market overview product",
    "news_article":    "news article",
    "archived_pdf":    "historical report pdf 2018 2019 2020",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _truncate(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", text)
    return " ".join(words[:max_words])


def _ddg_search(query: str, max_results: int = CANDIDATES_PER_DOC) -> list[str]:
    """Return up to max_results URLs from DuckDuckGo text search."""
    urls: list[str] = []
    try:
        with DDGS() as ddgs:
            for hit in ddgs.text(query, max_results=max_results):
                url = hit.get("href") or hit.get("url", "")
                if url and url.startswith("http"):
                    urls.append(url)
    except Exception as exc:
        print(f"    [DDG error] {exc}")
    return urls


def _fetch_best(urls: list[str]) -> dict[str, Any]:
    """Try each URL; return the first that yields >= MIN_WORDS of text."""
    for url in urls:
        # Skip obviously bad URLs
        if any(skip in url for skip in ("youtube.com", "twitter.com", "facebook.com", "linkedin.com")):
            continue
        result = fetch_document(url, timeout=FETCH_TIMEOUT)
        text = result.get("raw_text", "")
        wc = _word_count(text)
        if wc >= MIN_WORDS:
            return {
                "fetched_url": url,
                "word_count": wc,
                "status": "ok",
                "fetched_text": _truncate(text, MAX_WORDS),
            }
    return {"fetched_url": "", "word_count": 0, "status": "failed", "fetched_text": ""}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_existing(path: Path) -> set[str]:
    """Return doc_ids that were successfully fetched (status == ok)."""
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["doc_id"] for row in csv.DictReader(fh) if row.get("status") == "ok"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    resume = "--resume" in sys.argv

    documents = _load_csv(DATA_DIR / "documents.csv")
    queries_by_id = {q["query_id"]: q for q in _load_csv(DATA_DIR / "queries.csv")}
    existing = _load_existing(OUTPUT_PATH) if resume else set()

    if resume and existing:
        print(f"Resuming — {len(existing)} docs already fetched, skipping them.")

    # In resume mode, preserve only the successful rows then reopen for append.
    if resume and OUTPUT_PATH.exists() and existing:
        with OUTPUT_PATH.open(newline="", encoding="utf-8") as fh:
            kept = [r for r in csv.DictReader(fh) if r.get("status") == "ok"]
        with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
            w.writeheader()
            w.writerows(kept)
        print(f"  Kept {len(kept)} successful rows; will append new results.")

    mode = "a" if resume and OUTPUT_PATH.exists() else "w"
    out_fh = OUTPUT_PATH.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(out_fh, fieldnames=OUTPUT_FIELDS)
    if mode == "w":
        writer.writeheader()

    ok = failed = 0

    try:
        for i, doc in enumerate(documents):
            doc_id = doc["doc_id"]

            if doc_id in existing:
                continue

            query_id = doc.get("query_id", "")
            doc_type = doc.get("document_type", "")
            query_row = queries_by_id.get(query_id, {})
            topic = query_row.get("query", doc.get("title", ""))

            type_suffix = _TYPE_SUFFIX.get(doc_type, "report")
            search_query = f"{topic} {type_suffix}"

            print(f"[{i+1:>3}/100] {doc_id:<12} {doc_type:<18} searching: {search_query[:60]}")

            urls = _ddg_search(search_query)
            time.sleep(DDG_PAUSE)

            result = _fetch_best(urls)
            status = result["status"]
            wc = result["word_count"]

            if status == "ok":
                ok += 1
                print(f"           OK  url={result['fetched_url'][:60]}  words={wc}")
            else:
                failed += 1
                print(f"           FAILED (no usable URL from {len(urls)} candidates)")

            writer.writerow({
                "doc_id": doc_id,
                "query_id": query_id,
                "document_type": doc_type,
                "fetched_url": result["fetched_url"],
                "word_count": wc,
                "status": status,
                "fetched_text": result["fetched_text"],
            })
            out_fh.flush()

    finally:
        out_fh.close()

    total = ok + failed
    print(f"\nDone. {ok}/{total} documents fetched successfully, {failed} failed.")
    print(f"Output: {OUTPUT_PATH}")
    if failed:
        print(f"Re-run with --resume to retry only the failed ones (after fixing issues).")


if __name__ == "__main__":
    main()
