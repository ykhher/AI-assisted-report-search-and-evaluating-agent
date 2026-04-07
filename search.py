"""Search module with API-backed retrieval and safe local fallback."""

from __future__ import annotations

from urllib.parse import urlparse

import re

from API import search_market_reports
from duplicate import deduplicate
from parser import parse_search_results
from query_handler import generate_queries


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

def expand_query(query: str) -> str:
    """Append standard suffixes to bias search results toward formal reports."""
    base_query = str(query or "").strip()
    required_terms = ["pdf", "report", "2024", "industry", "analysis"]
    missing_terms = [term for term in required_terms if term not in base_query.lower()]
    return f"{base_query} {' '.join(missing_terms)}".strip()


# ---------------------------------------------------------------------------
# Mock search backend (fallback when API is unavailable)
# ---------------------------------------------------------------------------

_MOCK_INDEX = [
    {
        "title": "Global AI Adoption in Enterprise: 2024 Industry Report",
        "url": "https://research.example.org/ai-enterprise-2024.pdf",
        "snippet": (
            "This report presents a comprehensive methodology for evaluating AI adoption "
            "across Fortune 500 companies. References include 87 peer-reviewed studies. "
            "Key findings: 63% of enterprises increased AI budgets by 2.4x in 2024. "
            "Methodology section describes a stratified random sample of 1,200 firms. "
            "Introduction outlines scope; Conclusion summarizes 14 policy recommendations. "
            "Statistical appendix contains 340 data tables covering 42 industries. "
            "Bibliography lists 112 sources including WHO, McKinsey, and Stanford HAI. "
            "Word count estimate: 18,000 words across 7 structured sections."
        ),
        "source": "research.example.org",
        "is_pdf": True,
    },
    {
        "title": "Renewable Energy Market Outlook 2024",
        "url": "https://energyinstitute.example.com/reports/renewable-2024.pdf",
        "snippet": (
            "Published by the Global Energy Institute. Methodology: longitudinal analysis "
            "of 15-year energy transition data across 90 countries. This report references "
            "IPCC, IEA, and IRENA datasets. Contains 5,200 numeric data points across "
            "wind, solar, and hydro segments. Introduction, Literature Review, Methodology, "
            "Results, and Conclusion sections present findings on 2.7 TW of new capacity. "
            "Bibliography includes 95 citations from government and academic sources."
        ),
        "source": "energyinstitute.example.com",
        "is_pdf": True,
    },
    {
        "title": "5 AI Trends You Need to Know in 2024",
        "url": "https://techblog.example.io/ai-trends-2024",
        "snippet": (
            "A quick roundup of the hottest AI trends this year. GPT-4 is amazing. "
            "Check out these cool tools. Subscribe for more updates every week. "
            "This is a short listicle written for SEO purposes."
        ),
        "source": "techblog.example.io",
        "is_pdf": False,
    },
    {
        "title": "Healthcare AI: Benchmarking Clinical Decision Support Systems",
        "url": "https://medresearch.example.net/cds-benchmark-2024.pdf",
        "snippet": (
            "Systematic review and benchmarking report. Methodology: PRISMA framework "
            "applied to 230 clinical trials from PubMed. Introduction describes diagnostic "
            "accuracy baselines. References section cites 148 peer-reviewed journal articles. "
            "Quantitative results: AUC scores ranging from 0.71 to 0.94 across 12 systems. "
            "Contains statistical analysis of 14,000 patient records across 6 hospital networks. "
            "Conclusion recommends 3-tier regulatory framework for clinical AI deployment."
        ),
        "source": "medresearch.example.net",
        "is_pdf": True,
    },
    {
        "title": "Supply Chain Resilience Post-Pandemic: A Corporate Survey",
        "url": "https://logistics.example.com/supply-chain-report-2024.pdf",
        "snippet": (
            "Survey-based report covering 800 procurement executives across 35 countries. "
            "Methodology section describes stratified sampling and Likert-scale instrument design. "
            "Introduction contextualizes COVID-19 disruption impacts on global logistics. "
            "References 62 industry whitepapers and government trade statistics. "
            "Key metrics: 47% of firms increased buffer inventory by 3.1 months. "
            "Conclusion identifies 9 strategic resilience levers. 12,400 words total."
        ),
        "source": "logistics.example.com",
        "is_pdf": True,
    },
    {
        "title": "Quick Guide to Writing Better Supply Chain Emails",
        "url": "https://supplychainblog.example.com/emails-guide",
        "snippet": (
            "Boost your supply chain communication. Here are 7 email templates. "
            "Written by a content marketing intern. No references or data included."
        ),
        "source": "supplychainblog.example.com",
        "is_pdf": False,
    },
    {
        "title": "Cybersecurity Threat Landscape 2024: CISO Perspective",
        "url": "https://securityfoundation.example.org/threat-report-2024.pdf",
        "snippet": (
            "Annual threat intelligence report from the Global Security Foundation. "
            "Methodology: telemetry from 4.2 billion security events across 190 countries. "
            "Introduction frames the evolving ransomware ecosystem. Bibliography cites 74 CVEs, "
            "NIST frameworks, and MITRE ATT&CK documentation. Contains 2,800 numeric indicators "
            "of compromise (IoCs). Conclusion outlines zero-trust implementation roadmap. "
            "Structured into 8 chapters: Threat Actors, Vulnerabilities, Incident Response, "
            "Compliance, AI in Security, Cloud Risks, OT/ICS, and Recommendations."
        ),
        "source": "securityfoundation.example.org",
        "is_pdf": True,
    },
]


def _normalize_result(item: dict) -> dict:
    """Normalize raw API items to the internal search result schema."""
    url = str(item.get("url") or "")
    source = item.get("source") or item.get("domain") or urlparse(url).netloc

    return {
        "title": item.get("title") or item.get("name") or "Untitled result",
        "url": url,
        "snippet": item.get("snippet") or item.get("description") or "",
        "source": source,
        "date": item.get("date"),
        "year": item.get("year"),
        "is_pdf": url.lower().endswith(".pdf") or "pdf" in url.lower(),
    }


def _topic_terms(query: str) -> set[str]:
    """Extract the user-specific query words and ignore generic search boilerplate."""
    generic_terms = {
        "pdf", "report", "reports", "2024", "2025", "industry", "analysis",
        "market", "forecast", "outlook", "cagr", "projection", "research",
        "size", "revenue", "filetype",
    }
    terms = {
        term for term in re.findall(r"[a-z0-9]+", str(query).lower())
        if len(term) > 2 and term not in generic_terms
    }
    return terms or {
        term for term in re.findall(r"[a-z0-9]+", str(query).lower()) if len(term) > 2
    }


def _mock_search_reports(query: str, count: int = 10) -> list[dict]:
    """Local keyword-overlap fallback used when the API is unavailable."""
    keywords = _topic_terms(query)
    scored = []

    for doc in _MOCK_INDEX:
        doc_text = f"{doc['title']} {doc['snippet']}".lower()
        overlap = sum(1 for kw in keywords if kw in doc_text)
        if overlap > 0:
            scored.append((overlap, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:count]]


def _keyword_overlap_score(result: dict, query: str) -> int:
    """Score a result by topic-word overlap for stable local ranking."""
    query_terms = _topic_terms(query)
    text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
    return sum(1 for term in query_terms if term in text)


def search_once(query: str, count: int = 10) -> list[dict]:
    """Call the real search API exactly once for a query and return normalized results."""
    expanded_query = expand_query(query)
    try:
        print(f"[search_once] Calling real API once for: {expanded_query}")
        raw_response = search_market_reports(expanded_query, count=count)
        parsed_results = parse_search_results(raw_response)
        normalized_results = [_normalize_result(item) for item in parsed_results if isinstance(item, dict)]
        unique_results = deduplicate(normalized_results)
        unique_results.sort(key=lambda item: _keyword_overlap_score(item, query), reverse=True)
        return unique_results[:count]
    except Exception as exc:
        print(f"[search_once] API request failed: {exc}")
        return []


def search_reports(query: str, count: int = 10, use_api: bool = True) -> list[dict]:
    """Retrieve candidate report documents using API search plus local fallbacks."""
    if use_api:
        api_results = search_once(query, count=count)
        if api_results:
            return api_results
        print("[search] Falling back to local mock data after API failure.")

    query_variants = [expand_query(query), *generate_queries(query)]
    aggregated_results: list[dict] = []

    for search_query in query_variants:
        aggregated_results.extend(_mock_search_reports(search_query, count=count))

    unique_results = deduplicate(aggregated_results)
    unique_results.sort(key=lambda item: _keyword_overlap_score(item, query), reverse=True)

    focused_results = [item for item in unique_results if _keyword_overlap_score(item, query) > 0]
    return (focused_results or unique_results)[:count]
