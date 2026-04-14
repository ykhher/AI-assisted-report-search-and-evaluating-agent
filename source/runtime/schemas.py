"""Dataclasses for parsed documents and ranked report results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ParsedDocument:
    """Metadata extracted from full-text parsing."""

    word_count: int = 0
    has_methodology: bool = False
    has_references: bool = False
    has_statistics_language: bool = False
    section_lengths: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParsedDocument":
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class ScoreBreakdown:
    """Component scores used to rank one report."""

    relevance_score: float = 0.0
    report_validity_score: float = 0.0
    quality_score: float = 0.0
    authority_score: float = 0.0
    final_score: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoreBreakdown":
        """Create instance from dictionary."""
        return cls(**data)

    def avg_score(self) -> float:
        """Return average of the four component scores, excluding final."""
        scores = [
            self.relevance_score,
            self.report_validity_score,
            self.quality_score,
            self.authority_score,
        ]
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class RankedReport:
    """Final ranked report shape returned by the pipeline."""

    query: str = ""
    title: str = ""
    url: str = ""
    year: int | None = None
    source: str = ""
    source_class: str = "unknown"
    authority_prior: float = 0.45
    report_type: str = "unknown"
    report_validity_score_classifier: float = 0.0
    relevance_score: float = 0.0
    report_validity_score: float = 0.0
    quality_score: float = 0.0
    authority_score: float = 0.0
    final_score: float = 0.0
    score_breakdown: ScoreBreakdown | None = None
    top_signals: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    warnings: list[str] = field(default_factory=list)
    index: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.score_breakdown:
            data["score_breakdown"] = self.score_breakdown.to_dict()
        return data

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RankedReport":
        """Create instance from dictionary."""
        payload = dict(data)
        breakdown = payload.get("score_breakdown")
        if isinstance(breakdown, dict):
            payload["score_breakdown"] = ScoreBreakdown.from_dict(breakdown)
        return cls(**payload)

    def quality_assessment(self) -> str:
        """Return a brief quality assessment based on scores."""
        avg = (
            self.relevance_score
            + self.report_validity_score
            + self.quality_score
            + self.authority_score
        ) / 4.0
        if avg >= 0.75:
            return "high"
        if avg >= 0.50:
            return "medium"
        return "low"


@dataclass
class BatchResults:
    """Container for all ranked results for one query."""

    query: str
    results: list[RankedReport] = field(default_factory=list)
    total_count: int = 0
    returned_count: int = 0
    iteration_count: int = 0
    failure_type: str | None = None
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "results": [result.to_dict() for result in self.results],
            "total_count": self.total_count,
            "returned_count": self.returned_count,
            "iteration_count": self.iteration_count,
            "failure_type": self.failure_type,
            "processing_time_ms": self.processing_time_ms,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchResults":
        """Create instance from dictionary."""
        return cls(
            query=data["query"],
            results=[RankedReport.from_dict(item) for item in data.get("results", [])],
            total_count=data.get("total_count", 0),
            returned_count=data.get("returned_count", 0),
            iteration_count=data.get("iteration_count", 0),
            failure_type=data.get("failure_type"),
            processing_time_ms=data.get("processing_time_ms", 0.0),
        )

    def avg_score(self) -> float:
        """Average final score across all results."""
        if not self.results:
            return 0.0
        return sum(result.final_score for result in self.results) / len(self.results)

    def top_result(self) -> RankedReport | None:
        """Return the top-ranked result, or None if empty."""
        return self.results[0] if self.results else None
