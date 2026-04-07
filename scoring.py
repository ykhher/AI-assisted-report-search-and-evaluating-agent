"""Phase 4 scoring and ranking for report credibility."""

from __future__ import annotations

import json
import re
from typing import Any

_SECTION_KEYWORDS = ("introduction", "methodology", "results", "conclusion")
_CLAIM_KEYWORDS = ("increase", "decrease", "forecast", "projected", "expected")
_CITATION_BRACKET_PATTERN = re.compile(r"\[\d+\]")
_CITATION_YEAR_PATTERN = re.compile(r"\((?:19|20)\d{2}\)")
_NUMBERED_REFERENCE_PATTERN = re.compile(r"(?:^|\n)\s*\d+\.\s", re.MULTILINE)
_REFERENCE_SECTION_KEYWORDS = ("references", "bibliography", "sources")
_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")


def _clamp01(value: Any) -> float:
    """Convert a value to float and clamp it to the range [0, 1]."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(numeric, 1.0))


def _word_count(text: str) -> int:
    """Count words in a text string."""
    return len(re.findall(r"\b\w+\b", str(text).lower()))


def compute_structure_score(text: str) -> float:
    """Measure whether the report contains standard structural sections."""
    cleaned = str(text).lower()
    detected_sections = sum(1 for section in _SECTION_KEYWORDS if section in cleaned)
    return round(detected_sections / len(_SECTION_KEYWORDS), 3)


def compute_claim_density(text: str) -> float:
    """Measure how densely the text uses analytical claim language."""
    cleaned = str(text).lower()
    total_words = _word_count(cleaned)
    if total_words == 0:
        return 0.0

    keyword_count = sum(
        len(re.findall(fr"\b{re.escape(keyword)}\b", cleaned))
        for keyword in _CLAIM_KEYWORDS
    )
    return round(min((keyword_count / total_words) / 0.01, 1.0), 3)


def in_text_citation_score(text: str) -> float:
    """Detect academic-style in-text citations like `[1]` and `(2024)`."""
    raw_text = str(text)
    count = (
        len(_CITATION_BRACKET_PATTERN.findall(raw_text))
        + len(_CITATION_YEAR_PATTERN.findall(raw_text))
    )
    return round(min(count / 20, 1.0), 3)


def reference_section_score(text: str) -> float:
    """Detect a reference section and count lines that look like references."""
    raw_text = str(text)
    lowered = raw_text.lower()
    section_positions = [lowered.rfind(keyword) for keyword in _REFERENCE_SECTION_KEYWORDS if keyword in lowered]
    if not section_positions:
        return 0.0

    section_start = max(section_positions)
    section_text = raw_text[section_start:]
    lines = [line.strip() for line in section_text.splitlines() if line.strip()]
    if len(lines) <= 1:
        lines = [segment.strip() for segment in re.split(r"(?<=[.;])\s+", section_text) if segment.strip()]

    reference_lines = 0
    for line in lines[1:]:
        word_count = len(re.findall(r"\b\w+\b", line))
        if "http" in line.lower() or "www" in line.lower() or word_count > 8:
            reference_lines += 1

    return round(min(reference_lines / 20, 1.0), 3)


def numbered_reference_score(text: str) -> float:
    """Detect numbered reference lines like `1. McKinsey (2023)`."""
    count = len(_NUMBERED_REFERENCE_PATTERN.findall(str(text)))
    return round(min(count / 15, 1.0), 3)


def compute_citation_score(text: str) -> float:
    """Combine academic and industry-style reference evidence into one citation score."""
    base_score = max(in_text_citation_score(text), reference_section_score(text))
    boosted_score = base_score + 0.3 * numbered_reference_score(text)
    return round(min(boosted_score, 1.0), 3)


def compute_consistency_score(text: str) -> float:
    """Measure whether detected sections also contain numbers or analytical claims."""
    lowered = str(text).lower()
    detected_sections = [section for section in _SECTION_KEYWORDS if section in lowered]
    if not detected_sections:
        return 0.0

    valid_sections = 0
    for section in detected_sections:
        start = lowered.find(section)
        next_positions = [lowered.find(other, start + 1) for other in _SECTION_KEYWORDS if lowered.find(other, start + 1) != -1]
        end = min(next_positions) if next_positions else len(lowered)
        segment = lowered[start:end]

        has_number = bool(_NUMBER_PATTERN.search(segment))
        has_claim = any(keyword in segment for keyword in _CLAIM_KEYWORDS)
        if has_number or has_claim:
            valid_sections += 1

    return round(valid_sections / len(detected_sections), 3)


def compute_rqi(signals: dict, text: str = "") -> float:
    """Compute the upgraded Report Quality Index (RQI) with stronger score separation."""
    raw_text = text or str(signals.get("_text", ""))

    structure_score = _clamp01(signals.get("structure_score", compute_structure_score(raw_text)))
    claim_density = _clamp01(signals.get("claim_density", compute_claim_density(raw_text)))
    citation_score = _clamp01(signals.get("citation_score", compute_citation_score(raw_text)))
    consistency_score = _clamp01(signals.get("consistency_score", compute_consistency_score(raw_text)))

    signals["structure_score"] = structure_score
    signals["claim_density"] = claim_density
    signals["citation_score"] = citation_score
    signals["consistency_score"] = consistency_score

    methodology = _clamp01(signals.get("methodology", signals.get("has_methodology", 0)))
    data_density = _clamp01(signals.get("data_density", 0))
    source = _clamp01(signals.get("source", signals.get("source_reputation", 0)))
    recency = _clamp01(signals.get("recency", 0))
    data_component = data_density ** 1.3

    signals["data_component"] = round(data_component, 3)

    rqi = (
        0.12 * methodology
        + 0.22 * citation_score
        + 0.15 * data_component
        + 0.15 * source
        + 0.10 * recency
        + 0.13 * structure_score
        + 0.08 * claim_density
    )

    selective_boost = methodology == 1.0 and citation_score > 0.6
    confidence_boost = (
        methodology == 1.0
        and citation_score > 0.5
        and structure_score > 0.7
    )

    if selective_boost:
        rqi += 0.05
    if confidence_boost:
        rqi += 0.05

    signals["selective_boost"] = float(selective_boost)
    signals["confidence_boost"] = float(confidence_boost)

    return round(min(_clamp01(rqi), 1.0), 3)


def final_score(relevance: float, rqi: float, alpha: float = 0.55) -> float:
    """Blend relevance and credibility into a final ranking score, favoring topical match."""
    alpha = _clamp01(alpha)
    score = alpha * _clamp01(relevance) + (1 - alpha) * _clamp01(rqi)
    return round(_clamp01(score), 3)


def generate_reason(signals: dict, relevance: float | None = None) -> str:
    """Generate a deterministic explanation string from signal and relevance rules."""
    reasons: list[str] = []

    if _clamp01(signals.get("methodology", signals.get("has_methodology", 0))) == 1.0:
        reasons.append("contains methodology")

    if _clamp01(signals.get("citation_score", 0)) >= 0.3:
        reasons.append("strong citation support")
    elif _clamp01(signals.get("references", signals.get("has_references", 0))) == 1.0:
        reasons.append("includes references")

    if _clamp01(signals.get("data_component", 0)) > 0.55:
        reasons.append("strong quantitative evidence")
    elif _clamp01(signals.get("data_density", 0)) > 0.5:
        reasons.append("strong statistical support")

    if _clamp01(signals.get("structure_score", 0)) >= 0.75:
        reasons.append("well-structured report")

    if _clamp01(signals.get("claim_density", 0)) >= 0.5:
        reasons.append("contains strong analytical claims")

    if _clamp01(signals.get("consistency_score", 0)) >= 0.7:
        reasons.append("internally consistent analysis")

    if _clamp01(signals.get("selective_boost", 0)) >= 1.0:
        reasons.append("high-confidence methodology and citations")

    if _clamp01(signals.get("confidence_boost", 0)) >= 1.0:
        reasons.append("overall high-quality report")

    if _clamp01(signals.get("recency", 0)) > 0.7:
        reasons.append("recent publication")

    if relevance is not None and _clamp01(relevance) < 0.3:
        reasons.append("low relevance to the query, try adding more details to the query")

    if reasons:
        return ", ".join(reasons)
    return "limited credibility indicators, try adding more detail to the query"


def build_reason(signals: dict, rqi: float | None = None, relevance: float | None = None) -> str:
    """Compatibility wrapper for the existing pipeline import."""
    return generate_reason(signals, relevance=relevance)


def rank_reports(reports: list) -> list:
    """Compute RQI, final score, and explanation for each report, then rank them."""
    ranked = []

    for report in reports:
        signals = dict(report.get("signals", {}) or {})
        relevance = report.get("relevance", 0)
        text = report.get("text") or signals.get("_text", "")
        rqi = compute_rqi(signals, text)
        score = final_score(relevance, rqi)
        reason = generate_reason(signals, relevance=relevance)

        ranked.append({
            "title": report.get("title", ""),
            "url": report.get("url", ""),
            "RQI": rqi,
            "score": score,
            "reason": reason,
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


if __name__ == "__main__":
    from extractor import extract_signals, source_score

    high_quality_text = ((
        "Introduction: This report evaluates the sector in detail. "
        "Methodology: We used a structured framework with 120 interviews and 48 case studies. "
        "Results: Revenue is expected to increase by 18% and projected demand may decrease in only 2 regions. "
        "Conclusion: The findings support a strong forecast outlook for 2024 and 2025. "
        "References [1] [2] [3] (2024) (2023). "
    ) * 70)

    medium_quality_text = (
        "Introduction to the market review. Methodology includes a survey of 75 firms. "
        "Results show an expected increase in adoption next year. "
        "Conclusion highlights a forecast trend. References (2024). "
    )

    low_quality_text = ((
        "AI market AI market AI market forecast forecast forecast increase increase expected expected. "
        "This content repeats the same buzzwords again and again with little substance and no real citations. "
    ) * 40)

    test_cases = [
        {
            "name": "High-quality report",
            "text": high_quality_text,
            "metadata": {"year": 2024, "source": "research.example.org"},
            "relevance": 0.9,
        },
        {
            "name": "Medium-quality report",
            "text": medium_quality_text,
            "metadata": {"year": 2024, "source": "industry.example.com"},
            "relevance": 0.6,
        },
        {
            "name": "Low-quality blog-like content",
            "text": low_quality_text,
            "metadata": {"year": 2022, "source": "blog.example.io"},
            "relevance": 0.3,
        },
    ]

    for case in test_cases:
        signals = extract_signals(case["text"], case["metadata"])
        signals["source"] = source_score(case["metadata"]["source"])
        signals["_text"] = case["text"]

        rqi = compute_rqi(signals, case["text"])
        score = final_score(case["relevance"], rqi)
        reason = generate_reason(signals, relevance=case["relevance"])
        printable_signals = {key: value for key, value in signals.items() if not key.startswith("_")}

        print("\n" + "=" * 70)
        print(case["name"])
        print("Signals:")
        print(json.dumps(printable_signals, indent=2))
        print(f"RQI: {rqi}")
        print(f"Final score: {score}")
        print(f"Explanation: {reason}")
