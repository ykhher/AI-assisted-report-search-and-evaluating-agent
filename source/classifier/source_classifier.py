"""Classify a source by authority using URL, title, and text hints."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse


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
    ".org": "research_institute",
}

KEYWORD_GROUPS: tuple[tuple[str, set[str]], ...] = (
    ("government", GOVERNMENT_KEYWORDS),
    ("academic", ACADEMIC_KEYWORDS),
    ("consulting", CONSULTING_KEYWORDS),
    ("research_institute", RESEARCH_INSTITUTE_KEYWORDS),
    ("news_media", NEWS_MEDIA_KEYWORDS),
    ("company_whitepaper", COMPANY_WHITEPAPER_KEYWORDS),
    ("marketing_site", MARKETING_SITE_KEYWORDS),
)

AUTHORITY_PRIORS: dict[str, float] = {
    "government": 0.95,
    "academic": 0.90,
    "research_institute": 0.85,
    "consulting": 0.75,
    "company_whitepaper": 0.70,
    "news_media": 0.65,
    "marketing_site": 0.35,
    "unknown": 0.45,
}


def _normalize_text(text: str) -> str:
    """Normalize text for matching."""
    return " ".join(str(text or "").lower().split())


def _match_keywords(text: str, keywords: set[str]) -> bool:
    """Return True when any keyword appears in text."""
    return any(keyword in text for keyword in keywords)


def _get_domain(url: str) -> str:
    """Extract a lowercase domain from a URL-like string."""
    try:
        parsed = urlparse(url)
        return (parsed.netloc or parsed.path).lower().split("/")[0]
    except Exception:
        return ""


def _classify_by_tld(url: str) -> str | None:
    """Classify quickly using domain suffix hints."""
    domain = _get_domain(url)
    for suffix, label in TLD_PATTERNS.items():
        if domain.endswith(suffix):
            return label
    return None


def _classify_by_domain_keywords(url: str, text: str = "") -> str | None:
    """Classify from domain/text keyword groups in priority order."""
    combined = _normalize_text(f"{url} {text}")
    for label, keywords in KEYWORD_GROUPS:
        if _match_keywords(combined, keywords):
            return label
    return None


def _classify_by_title_hints(title: str) -> str | None:
    """Extract lightweight source hints from document title."""
    normalized_title = _normalize_text(title)

    if "government" in normalized_title or "official" in normalized_title:
        return "government"
    if "university" in normalized_title:
        return "academic"
    if "white paper" in normalized_title or "whitepaper" in normalized_title:
        return "company_whitepaper"
    if "research" in normalized_title or "analysis" in normalized_title:
        return "consulting" if _match_keywords(normalized_title, CONSULTING_KEYWORDS) else "research_institute"
    if _match_keywords(normalized_title, NEWS_MEDIA_KEYWORDS):
        return "news_media"
    return None


def classify_source(url: str, title: str = "", text: str = "") -> dict[str, Any]:
    """Classify a source and return its authority prior."""
    classification = (
        _classify_by_domain_keywords(url, text)
        or _classify_by_title_hints(title)
        or _classify_by_tld(url)
        or "unknown"
    )
    return {
        "source_class": classification,
        "authority_prior": AUTHORITY_PRIORS.get(classification, AUTHORITY_PRIORS["unknown"]),
    }
