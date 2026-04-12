"""Classify a source by authority using URL, title, and text hints."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse
GOVERNMENT_DOMAINS = {
    ".gov",
    ".edu.gov",
    ".ac.uk",
}

GOVERNMENT_KEYWORDS = {
    "worldbank",
    "world bank",
    "imf.org",
    "oecd",
    "un.org",
    "fao.org",
    "wto.org",
    "european commission",
    "whitehouse",
    "congress.gov",
    "parliament",
    "senate.gov",
    "house.gov",
    "sec.gov",
    "cdc.gov",
    "epa.gov",
    "noaa.gov",
}

ACADEMIC_DOMAINS = {
    ".edu",
    ".ac.uk",
    ".edu.au",
    ".ac.nz",
    ".ac.jp",
}

ACADEMIC_KEYWORDS = {
    "arxiv.org",
    "scholar.google",
    "researchgate",
    "academia.edu",
    "oxford",
    "cambridge",
    "harvard",
    "mit.edu",
    "stanford.edu",
    "berkeley.edu",
    "caltech",
    "yale",
    "princeton",
    ".university",
}

RESEARCH_INSTITUTE_KEYWORDS = {
    "brookings",
    "rand.org",
    "csis.org",
    "cfr.org",
    "heritage.org",
    "aei.org",
    "gallup",
    "pew research",
    "statista",
    "eurostat",
    "census.gov",
    "bureau of labor",
    "federal reserve",
    "energy information",
    "international energy agency",
    "iea.org",
    "ipcc",
}

CONSULTING_KEYWORDS = {
    "mckinsey",
    "goldman sachs",
    "bcg.com",
    "bain & company",
    "deloitte",
    "pwc",
    "kpmg",
    "accenture",
    "capgemini",
    "ey.com",
    "gartner",
    "forrester",
    "idg",
    "canalyst",
}

COMPANY_WHITEPAPER_KEYWORDS = {
    "white paper",
    "whitepaper",
    "case study",
    "market report",
    "industry report",
    "analyst report",
}

NEWS_MEDIA_KEYWORDS = {
    "reuters",
    "bloomberg",
    "financial times",
    "ft.com",
    "wsj.com",
    "economist",
    "bbc",
    "cnn",
    "associated press",
    "ap news",
    "cnbc",
    "thehill",
}

MARKETING_SITE_KEYWORDS = {
    "blog",
    "medium.com",
    ".io",
    ".app",
    "startup",
    "tutorial",
    "guide",
    "tips",
    "guide to",
    "how to",
}

TLD_PATTERNS = {
    ".gov": "government",
    ".edu": "academic",
    ".mil": "government",
    ".ac.uk": "academic",
    ".org": "research_institute",  # will be refined by domain keywords
}

AUTHORITY_PRIORS: dict[str, float] = {
    "government": 0.95,
    "academic": 0.90,
    "research_institute": 0.85,
    "consulting": 0.75,
    "company_whitepaper": 0.60,
    "news_media": 0.65,
    "marketing_site": 0.35,
    "unknown": 0.45,
}

def _normalize_text(text: str) -> str:
    """Normalize text for matching."""
    return " ".join(str(text or "").lower().split())


def _match_keywords(text: str, keywords: set[str]) -> bool:
    """Return True when any keyword appears in text."""
    normalized = _normalize_text(text)
    return any(keyword in normalized for keyword in keywords)


def _get_domain(url: str) -> str:
    """Extract domain from a URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        domain = domain.lower().split("/")[0]
        return domain
    except Exception:
        return ""


def _get_tld(domain: str) -> str:
    """Extract a known TLD from domain."""
    domain_lower = domain.lower()
    for pattern in TLD_PATTERNS:
        if domain_lower.endswith(pattern):
            return pattern
    return ""


def _classify_by_tld(url: str) -> str | None:
    """Classify quickly using TLD hints."""
    domain = _get_domain(url)
    tld = _get_tld(domain)
    if tld:
        return TLD_PATTERNS.get(tld)
    return None


def _classify_by_domain_keywords(url: str, text: str = "") -> str | None:
    """Classify by direct domain and text keyword matching."""
    combined = _normalize_text(f"{url} {text}")

    if _match_keywords(combined, GOVERNMENT_KEYWORDS):
        return "government"

    if _match_keywords(combined, ACADEMIC_KEYWORDS):
        return "academic"

    if _match_keywords(combined, CONSULTING_KEYWORDS):
        return "consulting"

    if _match_keywords(combined, RESEARCH_INSTITUTE_KEYWORDS):
        return "research_institute"

    if _match_keywords(combined, NEWS_MEDIA_KEYWORDS):
        return "news_media"

    if _match_keywords(combined, COMPANY_WHITEPAPER_KEYWORDS):
        return "company_whitepaper"

    if _match_keywords(combined, MARKETING_SITE_KEYWORDS):
        return "marketing_site"

    return None


def _classify_by_title_hints(title: str) -> str | None:
    """Extract classification hints from document title."""
    normalized_title = _normalize_text(title)

    if "government" in normalized_title or "official" in normalized_title:
        return "government"

    if "university" in normalized_title:
        return "academic"

    if "white paper" in normalized_title or "whitepaper" in normalized_title:
        return "company_whitepaper"

    if "research" in normalized_title or "analysis" in normalized_title:
        if _match_keywords(normalized_title, CONSULTING_KEYWORDS):
            return "consulting"
        return "research_institute"

    if _match_keywords(normalized_title, NEWS_MEDIA_KEYWORDS):
        return "news_media"

    return None


def classify_source(url: str, title: str = "", text: str = "") -> dict[str, Any]:
    """Classify source and return source_class with authority_prior."""
    classification = (
        _classify_by_domain_keywords(url, text)
        or _classify_by_title_hints(title)
        or _classify_by_tld(url)
        or "unknown"
    )

    authority_prior = AUTHORITY_PRIORS.get(classification, AUTHORITY_PRIORS["unknown"])

    return {
        "source_class": classification,
        "authority_prior": authority_prior,
    }


if __name__ == "__main__":
    test_cases = [
        {
            "url": "https://www.worldbank.org/en/news/press-release",
            "title": "World Bank Annual Report 2024",
            "expected": "government",
        },
        {
            "url": "https://arxiv.org/abs/2401.12345",
            "title": "Deep Learning for Climate Modeling",
            "expected": "academic",
        },
        {
            "url": "https://www.mckinsey.com/industries/financial-services",
            "title": "McKinsey Global Banking Report 2024",
            "expected": "consulting",
        },
        {
            "url": "https://www.brookings.edu/research/",
            "title": "Brookings Institution Report",
            "expected": "research_institute",
        },
        {
            "url": "https://example.com/white-paper.pdf",
            "title": "Our Company's Cloud Platform White Paper",
            "expected": "company_whitepaper",
        },
        {
            "url": "https://www.reuters.com/business/",
            "title": "Reuters Financial News",
            "expected": "news_media",
        },
        {
            "url": "https://myblog.medium.com/best-ai-tools",
            "title": "5 AI Tools You Need to Know",
            "expected": "marketing_site",
        },
        {
            "url": "https://example-domain-12345.com/report",
            "title": "",
            "expected": "unknown",
        },
    ]

    print("\n" + "=" * 80)
    print("SOURCE CLASSIFICATION EXAMPLES")
    print("=" * 80)

    for i, case in enumerate(test_cases, 1):
        result = classify_source(case["url"], case["title"])
        match = "✓" if result["source_class"] == case["expected"] else "✗"

        print(f"\n[{i}] {match}")
        print(f"    URL: {case['url']}")
        print(f"    Title: {case['title']}")
        print(f"    Classified as: {result['source_class']}")
        print(f"    Authority Prior: {result['authority_prior']:.2f}")
        if result["source_class"] != case["expected"]:
            print(f"    Expected: {case['expected']}")

    print("\n" + "=" * 80)
