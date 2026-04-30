"""Rebuild curated benchmark documents using real SerpAPI search results.

For each of the 20 queries in queries.csv, calls search_market_reports() to
get 5 real documents with actual URLs, then fetches their text content.

Writes:
  data/curated_benchmark/documents.csv        — updated with real URLs/snippets
  data/curated_benchmark/document_texts.csv   — full fetched text per doc

Usage:
    SERPAPI_API_KEY=<key> python test/rebuild_benchmark_documents.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.API import search_market_reports
from source.fetching.document_fetcher import fetch_document
from source.fetching.parser import parse_search_results

DATA_DIR = PROJECT_ROOT / "data" / "curated_benchmark"
DOCS_PATH      = DATA_DIR / "documents.csv"
TEXTS_PATH     = DATA_DIR / "document_texts.csv"

RESULTS_PER_QUERY = 5
FETCH_TIMEOUT     = 15
MIN_WORDS         = 150
MAX_WORDS         = 2000
SEARCH_PAUSE      = 1.5   # seconds between SerpAPI calls

DOCS_FIELDS  = ["doc_id", "query_id", "retrieved_rank", "title", "source",
                "source_type", "url", "year", "document_type", "is_pdf", "snippet"]
TEXTS_FIELDS = ["doc_id", "query_id", "document_type", "fetched_url",
                "word_count", "status", "fetched_text"]

import re


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _truncate(text: str, max_words: int) -> str:
    return " ".join(re.findall(r"\S+", text)[:max_words])


def _infer_doc_type(result: dict) -> str:
    url   = str(result.get("url", "")).lower()
    title = str(result.get("title", "")).lower()
    snip  = str(result.get("snippet", "")).lower()
    combined = f"{url} {title} {snip}"

    if url.endswith(".pdf") or "filetype:pdf" in url:
        return "research_report"
    if any(w in combined for w in ("whitepaper", "white paper", "white-paper")):
        return "whitepaper"
    if any(w in combined for w in ("news", "press release", "announcement")):
        return "news_article"
    if any(w in combined for w in ("report", "research", "survey", "study", "benchmark", "outlook")):
        return "research_report"
    return "landing_page"


def _infer_source_type(url: str) -> str:
    domain = url.lower()
    if any(d in domain for d in (".gov", ".mil")):
        return "government"
    if any(d in domain for d in (".edu", "university", "institute")):
        return "research_institute"
    if any(d in domain for d in ("mckinsey", "bcg", "bain", "deloitte", "pwc", "ey.com", "kpmg")):
        return "consulting"
    if any(d in domain for d in (".org", "association", "federation", "council")):
        return "industry_association"
    if any(d in domain for d in ("reuters", "bloomberg", "ft.com", "wsj", "bbc", "cnbc")):
        return "news_media"
    return "vendor"


def _extract_year(text: str) -> str:
    m = re.search(r"\b(20[12]\d)\b", text)
    return m.group(1) if m else ""


def search_query(query_row: dict) -> list[dict]:
    query = query_row["query"]
    query_id = query_row["query_id"]
    print(f"\n[{query_id}] {query}")

    try:
        raw = search_market_reports(query, count=10)
        results = parse_search_results(raw)
    except Exception as exc:
        print(f"  SEARCH ERROR: {exc}")
        return []

    time.sleep(SEARCH_PAUSE)

    docs = []
    rank = 1
    for r in results:
        if rank > RESULTS_PER_QUERY:
            break
        url = r.get("url", "")
        if not url or not url.startswith("http"):
            continue
        if any(skip in url for skip in ("youtube.com", "twitter.com", "facebook.com", "linkedin.com")):
            continue

        doc_id = f"{query_id}_{rank:02d}"
        is_pdf = url.lower().endswith(".pdf")
        doc_type = _infer_doc_type(r)
        source_url = url
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        source_domain = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]

        docs.append({
            "doc_id": doc_id,
            "query_id": query_id,
            "retrieved_rank": rank,
            "title": title,
            "source": source_domain,
            "source_type": _infer_source_type(url),
            "url": source_url,
            "year": _extract_year(f"{title} {snippet}"),
            "document_type": doc_type,
            "is_pdf": str(is_pdf).lower(),
            "snippet": snippet[:300],
        })
        print(f"  [{rank}] {doc_type:<18} {url[:65]}")
        rank += 1

    return docs


def fetch_texts(docs: list[dict]) -> list[dict]:
    texts = []
    for doc in docs:
        url = doc["url"]
        result = fetch_document(url, timeout=FETCH_TIMEOUT)
        raw = result.get("raw_text", "")
        wc = _word_count(raw)
        status = "ok" if wc >= MIN_WORDS else "failed"
        texts.append({
            "doc_id": doc["doc_id"],
            "query_id": doc["query_id"],
            "document_type": doc["document_type"],
            "fetched_url": url,
            "word_count": wc,
            "status": status,
            "fetched_text": _truncate(raw, MAX_WORDS) if status == "ok" else "",
        })
        print(f"    fetch {doc['doc_id']}: {status} ({wc} words)")
    return texts


def main() -> None:
    import os
    if not os.environ.get("SERPAPI_API_KEY"):
        print("ERROR: SERPAPI_API_KEY not set.")
        sys.exit(1)

    with (DATA_DIR / "queries.csv").open(newline="", encoding="utf-8") as f:
        queries = list(csv.DictReader(f))

    print(f"Rebuilding benchmark for {len(queries)} queries × {RESULTS_PER_QUERY} docs each ...")

    all_docs: list[dict] = []
    all_texts: list[dict] = []

    for query_row in queries:
        docs = search_query(query_row)
        if not docs:
            print(f"  WARNING: no results for {query_row['query_id']}")
            continue
        texts = fetch_texts(docs)
        all_docs.extend(docs)
        all_texts.extend(texts)

    # Write documents.csv
    with DOCS_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DOCS_FIELDS)
        w.writeheader()
        w.writerows(all_docs)

    # Write document_texts.csv
    with TEXTS_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TEXTS_FIELDS)
        w.writeheader()
        w.writerows(all_texts)

    ok = sum(1 for t in all_texts if t["status"] == "ok")
    print(f"\nDone. {len(all_docs)} documents, {ok}/{len(all_texts)} fetched successfully.")
    print(f"  documents.csv     -> {DOCS_PATH}")
    print(f"  document_texts.csv -> {TEXTS_PATH}")


if __name__ == "__main__":
    main()
