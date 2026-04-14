"""Parse report-like text into clean sections and simple stats."""

from __future__ import annotations

import re

SECTION_KEYS: tuple[str, ...] = (
    "summary",
    "methodology",
    "results",
    "assumptions",
    "references",
)

# Map common heading variants to canonical keys.
SECTION_HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "summary": (
        "summary",
        "executive summary",
        "overview",
        "key highlights",
        "introduction",
    ),
    "methodology": (
        "methodology",
        "methods",
        "approach",
        "research design",
        "data collection",
        "sample design",
    ),
    "results": (
        "results",
        "findings",
        "analysis",
        "discussion",
        "market outlook",
        "projections",
    ),
    "assumptions": (
        "assumptions",
        "limitations",
        "scope and assumptions",
        "scenario assumptions",
        "caveats",
    ),
    "references": (
        "references",
        "bibliography",
        "sources",
        "source notes",
        "appendix references",
    ),
}

STATISTICAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d+(?:[.,]\d+)?%\b"),
    re.compile(r"\b(?:mean|median|std(?:\.\s*dev)?|variance)\b", re.IGNORECASE),
    re.compile(r"\b(?:sample size|n\s*=\s*\d+|confidence interval|p-value)\b", re.IGNORECASE),
    re.compile(r"\b(?:cagr|forecast|projection|growth rate|index)\b", re.IGNORECASE),
)

HEADER_ALIASES: tuple[tuple[str, str], ...] = tuple(
    (alias, canonical)
    for canonical, aliases in SECTION_HEADER_ALIASES.items()
    for alias in aliases
)


def clean_text(raw_text: str) -> str:
    """Normalize raw text while keeping paragraph breaks."""
    text = str(raw_text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n\t]+", " ", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_header_candidate(line: str) -> bool:
    """Return True when a line looks like a heading."""
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 90:
        return False
    if len(stripped.split()) > 10:
        return False
    if stripped.count(".") > 2:
        return False
    return True


def detect_section_header(line: str) -> str | None:
    """Detect a canonical section key from one line."""
    if not _is_header_candidate(line):
        return None

    normalized = line.strip().lower().rstrip(":")
    normalized = re.sub(r"^\s*\d+(?:[.)]|\s+-)\s*", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)

    for alias, canonical in HEADER_ALIASES:
        if normalized == alias:
            return canonical

    for alias, canonical in HEADER_ALIASES:
        if normalized.startswith(alias + " "):
            return canonical

    return None


def split_into_sections(cleaned_text: str) -> dict[str, str]:
    """Split cleaned text into canonical sections."""
    sections: dict[str, list[str]] = {key: [] for key in SECTION_KEYS}
    current_section = "summary"

    for line in cleaned_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        detected = detect_section_header(stripped)
        if detected is not None:
            current_section = detected
            continue

        sections[current_section].append(stripped)

    return {key: "\n".join(parts).strip() for key, parts in sections.items()}


def has_statistics_language(text: str) -> bool:
    """Return True when statistical or quantitative language is detected."""
    if not text:
        return False
    return any(pattern.search(text) is not None for pattern in STATISTICAL_PATTERNS)


def _word_count(text: str) -> int:
    """Count word-like tokens."""
    return len(re.findall(r"\b\w+\b", text))


def parse_report_text(raw_text: str) -> dict[str, object]:
    """Return cleaned text, parsed sections, and lightweight stats."""
    cleaned = clean_text(raw_text)
    sections = split_into_sections(cleaned)

    methodology_text = sections.get("methodology", "")
    references_text = sections.get("references", "")

    stats = {
        "word_count": _word_count(cleaned),
        "has_methodology": bool(methodology_text),
        "has_references": bool(references_text),
        "has_statistics_language": has_statistics_language(cleaned),
    }

    return {
        "clean_text": cleaned,
        "sections": {
            "summary": sections.get("summary", ""),
            "methodology": methodology_text,
            "results": sections.get("results", ""),
            "assumptions": sections.get("assumptions", ""),
            "references": references_text,
        },
        "stats": stats,
    }
