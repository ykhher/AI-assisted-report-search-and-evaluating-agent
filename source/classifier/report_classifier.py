"""Classify document type and estimate report validity."""

import re
from typing import Optional


REPORT_KEYWORDS = {
    "report", "analysis", "analytical", "assessment", "evaluation",
    "findings", "conclusion", "conclusions", "results", "outcome",
    "performance review", "annual report", "quarterly report", "status report"
}

WHITEPAPER_KEYWORDS = {
    "whitepaper", "white paper", "white-paper", "position paper",
    "technical paper", "research paper", "conceptual framework"
}

BENCHMARK_KEYWORDS = {
    "benchmark", "benchmarking", "comparison", "competitive analysis",
    "market sizing", "performance comparison", "study comparison"
}

SURVEY_KEYWORDS = {
    "survey", "questionnaire", "empirical study", "research study",
    "statistical analysis", "data analysis", "field study"
}

RESEARCH_NOTE_KEYWORDS = {
    "research note", "technical note", "note", "brief", "memo",
    "update", "insight", "technical brief"
}

DECK_KEYWORDS = {
    "presentation", "slides", "slide deck", "deck", "ppt",
    "powerpoint", "keynote", "presentation slides"
}

BROCHURE_KEYWORDS = {
    "brochure", "guide", "handbook", "manual", "directory",
    "reference guide", "field guide", "how-to", "tutorial"
}

BLOG_KEYWORDS = {
    "blog", "blog post", "post", "opinion", "commentary", "column",
    "thought leadership", "perspective", "editorial", "my thoughts"
}

LANDING_PAGE_KEYWORDS = {
    "landing page", "product page", "pricing page", "about us",
    "welcome page", "home page"
}

PROMOTIONAL_KEYWORDS = {
    "sign up", "free trial", "subscribe", "subscribe now",
    "buy now", "limited offer", "exclusive", "get started",
    "learn more", "contact us", "demo", "request demo",
    "schedule demo", "request trial"
}

SECTION_MARKERS = {
    r"^##\s+",
    r"^###\s+",
    r"^####\s+",
    r"^(Section|Chapter|Part)\s+\d+",
    r"^\d+\.\s+[A-Z]",
    r"^\s*-\s+[A-Z][a-z]+:",
}

REPORT_METHODOLOGY_KEYWORDS = {
    "methodology", "method", "approach", "we conducted", "we analyzed",
    "research design", "framework", "evaluation criteria", "data collection",
    "experimental design", "sample size"
}

REFERENCE_PATTERNS = [
    r"\[(\d+)\]",
    r"(\w+\s+et al\.?\s+\d{4})",
    r"(references|bibliography|citations)\s*:",
]

STRUCTURE_SECTION_KEYWORDS = {
    "executive summary", "introduction", "background", "methodology",
    "results", "findings", "analysis", "conclusion", "conclusions",
    "references", "appendix", "limitations", "discussion"
}


def _normalize_text(text: str) -> str:
    """Normalize text to lowercase for keyword matching."""
    return text.lower().strip() if text else ""


def _match_keywords(text: str, keyword_set: set) -> bool:
    """Check if any keyword from set appears in normalized text."""
    normalized = _normalize_text(text)
    return any(keyword in normalized for keyword in keyword_set)


def _count_section_markers(text: str) -> int:
    """Count approximate number of sections/subsections detected."""
    if not text:
        return 0
    count = 0
    for pattern in SECTION_MARKERS:
        count += len(re.findall(pattern, text, re.MULTILINE | re.IGNORECASE))
    return count


def _count_references(text: str) -> int:
    """Count approximate number of references/citations in text."""
    if not text:
        return 0
    count = 0
    for pattern in REFERENCE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        count += len(matches)
    return count


def _contains_report_structure(text: str) -> float:
    """Return a heuristic report-structure score in the range [0, 1]."""
    if not text:
        return 0.0

    normalized = _normalize_text(text)
    
    structure_keywords_found = sum(
        1 for kw in STRUCTURE_SECTION_KEYWORDS if kw in normalized
    )
    max_expected = len(STRUCTURE_SECTION_KEYWORDS)
    structure_score = min(structure_keywords_found / max_expected, 1.0) if max_expected > 0 else 0.0
    
    section_count = _count_section_markers(text)
    section_score = min(section_count / 10, 1.0)

    ref_count = _count_references(text)
    ref_score = min(ref_count / 5, 1.0)

    return (structure_score * 0.4 + section_score * 0.4 + ref_score * 0.2)


def _contains_methodology_language(text: str) -> bool:
    """Check if text contains research/report methodology indicators."""
    return _match_keywords(text, REPORT_METHODOLOGY_KEYWORDS)


def _is_promotional(title: str, text: str) -> bool:
    """Detect promotional/marketing language (call-to-action)."""
    combined = _normalize_text(title) + " " + _normalize_text(text[:500])
    return _match_keywords(combined, PROMOTIONAL_KEYWORDS)


def _is_pdf(metadata: Optional[dict]) -> bool:
    """Check if document is PDF format based on metadata."""
    if not metadata:
        return False
    fmt = metadata.get("format", "").lower()
    file_type = metadata.get("file_type", "").lower()
    return "pdf" in fmt or "pdf" in file_type


def classify_report_type(
    title: str = "",
    text: str = "",
    metadata: dict | None = None
) -> dict:
    """Classify document type and return a validity score."""

    normalized_title = _normalize_text(title)
    normalized_text = _normalize_text(text)

    # Catch clear call-to-action pages early.
    if _is_promotional(title, text):
        validity = 0.3 if _match_keywords(normalized_text, LANDING_PAGE_KEYWORDS) else 0.2
        return {
            "report_type": "landing_page",
            "report_validity_score": validity
        }

    # Title matches are checked in a specific-first order.
    if _match_keywords(normalized_title, WHITEPAPER_KEYWORDS):
        structure = _contains_report_structure(text)
        validity = 0.6 + (structure * 0.3)
        return {
            "report_type": "whitepaper",
            "report_validity_score": min(validity, 1.0)
        }
    
    if _match_keywords(normalized_title, BENCHMARK_KEYWORDS):
        has_methodology = _contains_methodology_language(text)
        validity = 0.65 + (0.2 if has_methodology else 0)
        return {
            "report_type": "benchmark",
            "report_validity_score": min(validity, 1.0)
        }
    
    if _match_keywords(normalized_title, DECK_KEYWORDS):
        validity = 0.7 if _is_pdf(metadata) else 0.6
        return {
            "report_type": "deck",
            "report_validity_score": validity
        }
    
    if _match_keywords(normalized_title, BROCHURE_KEYWORDS):
        return {
            "report_type": "brochure",
            "report_validity_score": 0.5
        }
    
    if _match_keywords(normalized_title, BLOG_KEYWORDS):
        return {
            "report_type": "blog",
            "report_validity_score": 0.4
        }
    
    if _match_keywords(normalized_title, RESEARCH_NOTE_KEYWORDS):
        has_methodology = _contains_methodology_language(text)
        validity = 0.55 + (0.2 if has_methodology else 0.05)
        return {
            "report_type": "research_note",
            "report_validity_score": min(validity, 1.0)
        }
    
    if _match_keywords(normalized_title, SURVEY_KEYWORDS):
        has_methodology = _contains_methodology_language(text)
        validity = 0.65 + (0.2 if has_methodology else 0.05)
        return {
            "report_type": "survey",
            "report_validity_score": min(validity, 1.0)
        }
    
    if _match_keywords(normalized_title, REPORT_KEYWORDS):
        structure = _contains_report_structure(text)
        validity = 0.70 + (structure * 0.25)
        return {
            "report_type": "report",
            "report_validity_score": min(validity, 1.0)
        }
    
    structure_score = _contains_report_structure(text)
    has_methodology = _contains_methodology_language(text)

    if structure_score > 0.6 and has_methodology:
        validity = 0.65 + (structure_score * 0.25)
        return {
            "report_type": "report",
            "report_validity_score": min(validity, 1.0)
        }
    
    if structure_score > 0.4 and not has_methodology:
        if _is_pdf(metadata):
            return {
                "report_type": "whitepaper",
                "report_validity_score": min(0.55 + structure_score * 0.20, 1.0)
            }
        return {
            "report_type": "research_note",
            "report_validity_score": min(0.50 + structure_score * 0.15, 1.0)
        }
    
    text_head = text[:500] if text else ""
    if text_head and _match_keywords(text_head, {"blog post", "blog", "article", "my thoughts", "opinion"}):
        return {
            "report_type": "blog",
            "report_validity_score": 0.45
        }
    
    fallback_validity = min(structure_score * 0.4, 0.5)
    return {
        "report_type": "unknown",
        "report_validity_score": fallback_validity
    }


# TEST

if __name__ == "__main__":
    test_cases = [
        {
            "title": "Q4 2024 Financial Performance Report",
            "text": """
            Executive Summary
            In Q4 2024, our company achieved strong results across all regions.
            
            Methodology
            We analyzed quarterly revenue and expense data using standard accounting practices.
            
            Section 1: Results
            - Revenue increased by 15% year-over-year
            - Operating margins improved by 2 percentage points
            
            Section 2: Conclusions
            The results indicate strong market demand and operational efficiency.
            
            References
            [1] Annual Financial Report 2023
            [2] Smith et al. 2024
            """,
            "metadata": {"format": "pdf", "word_count": 3000},
            "expected": "report"
        },
        {
            "title": "Blockchain Technology: A Position Whitepaper",
            "text": """
            Abstract
            This whitepaper outlines our technical approach to blockchain scalability.
            
            1. Introduction
            Blockchain technology has revolutionized distributed systems architecture.
            
            2. Methodology
            We designed a novel consensus mechanism based on Byzantine Fault Tolerance.
            
            3. Technical Results
            Our approach reduces transaction latency by 40% compared to Ethereum.
            
            References
            Nakamoto 2008, Ethereum Whitepaper 2013
            """,
            "metadata": {"format": "pdf"},
            "expected": "whitepaper"
        },
        {
            "title": "2024 Cloud Platform Benchmark Study",
            "text": """
            Executive Summary
            We benchmarked five leading cloud platforms.
            
            Methodology
            We conducted comparative testing across compute, storage, and networking layers.
            Methodology details: random sampling, 1000 test runs per platform.
            
            Findings
            - Platform A: 95ms latency, 99.99% availability
            - Platform B: 120ms latency, 99.95% availability
            
            Conclusion
            Platform A outperforms competitors on latency metrics.
            """,
            "metadata": {"format": "pdf"},
            "expected": "benchmark"
        },
        {
            "title": "Our Amazing SaaS Platform - Sign Up for Free",
            "text": """
            Welcome to our platform!
            Sign up now for a free trial!
            Limited offer - 50% off for first month!
            Buy now and transform your business.
            Contact us for a demo today.
            """,
            "metadata": {"format": "html"},
            "expected": "landing_page"
        },
        {
            "title": "AI Trends in 2024: My Thoughts and Observations",
            "text": """
            Blog Post - AI in 2024
            I've been thinking a lot about recent AI developments.
            Here are my personal thoughts on what's coming next.
            This is just my opinion, not a definitive analysis.
            """,
            "metadata": {},
            "expected": "blog"
        },
        {
            "title": "Quarterly Customer Satisfaction Survey 2024",
            "text": """
            Survey Overview
            We conducted a survey of 500 randomly selected customers.
            
            Methodology
            Random sampling with stratification by region and customer segment.
            Sample size: 500, Confidence level: 95%.
            
            Results
            Overall satisfaction: 4.2 out of 5.0
            Net Promoter Score: 42
            """,
            "metadata": {"format": "pdf"},
            "expected": "survey"
        },
        {
            "title": "Q3 2024 Business Results - Presentation Slides",
            "text": "Slide 1: Title Page\nSlide 2: Executive Overview\nSlide 3: Key Metrics\nSlide 4: Growth Trajectory",
            "metadata": {"format": "pdf", "file_type": "presentation"},
            "expected": "deck"
        },
        {
            "title": "Random Web Article About Technology",
            "text": "Some generic content that doesn't fit into clear analytical categories.",
            "metadata": {"format": "html"},
            "expected": "unknown"
        },
    ]
    
    print("=" * 80)
    print("REPORT TYPE CLASSIFICATION EXAMPLES")
    print("=" * 80)
    
    passed = 0
    for i, test in enumerate(test_cases, 1):
        result = classify_report_type(
            title=test["title"],
            text=test["text"],
            metadata=test["metadata"]
        )
        
        is_match = result["report_type"] == test["expected"]
        symbol = "Yes" if is_match else "No"
        if is_match:
            passed += 1
        
        print(f"\n[{i}] {symbol}")
        print(f"    Title: {test['title'][:55]}{'...' if len(test['title']) > 55 else ''}")
        print(f"    Type: {result['report_type']}")
        print(f"    Expected: {test['expected']}")
        print(f"    Validity Score: {result['report_validity_score']:.2f}")
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed}/{len(test_cases)} tests passed")
    print("=" * 80)
