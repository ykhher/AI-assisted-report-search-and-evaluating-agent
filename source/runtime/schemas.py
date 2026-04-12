"""Dataclasses for parsed documents and ranked report results."""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParsedDocument:
    """Metadata extracted from full-text parsing."""
    
    word_count: int = 0
    """Total word count of the document."""
    
    has_methodology: bool = False
    """Whether the document describes its research methodology."""
    
    has_references: bool = False
    """Whether the document includes bibliographic references."""
    
    has_statistics_language: bool = False
    """Whether the document contains statistical analysis language."""
    
    section_lengths: Dict[str, int] = field(default_factory=dict)
    """Section name -> word count."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParsedDocument":
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class ScoreBreakdown:
    """Component scores used to rank one report."""
    
    relevance_score: float = 0.0
    """Keyword overlap between query and document [0.0, 1.0]."""
    
    report_validity_score: float = 0.0
    """Likelihood document is a formal report/analysis [0.0, 1.0]."""
    
    quality_score: float = 0.0
    """Analytical quality and rigor indicators [0.0, 1.0]."""
    
    authority_score: float = 0.0
    """Source credibility and authority [0.0, 1.0]."""
    
    final_score: float = 0.0
    """Blended final ranking score [0.0, 1.0]."""
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoreBreakdown":
        """Create instance from dictionary."""
        return cls(**data)
    
    def avg_score(self) -> float:
        """Return average of the four component scores (excluding final)."""
        scores = [
            self.relevance_score,
            self.report_validity_score,
            self.quality_score,
            self.authority_score
        ]
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class RankedReport:
    """Final ranked report shape returned by the pipeline."""
    
    # Query context
    query: str = ""
    """Original search query that returned this result."""
    
    # Document identity
    title: str = ""
    """Document title."""
    
    url: str = ""
    """Document URL."""
    
    year: Optional[int] = None
    """Publication year (if parseable)."""
    
    # Source classification
    source: str = ""
    """Source domain/name (e.g., 'mckinsey.com')."""
    
    source_class: str = "unknown"
    """Source authority label."""
    
    authority_prior: float = 0.45
    """Base credibility score for the source category [0.0, 1.0]."""
    
    # Document classification
    report_type: str = "unknown"
    """Document type label."""
    
    report_validity_score_classifier: float = 0.0
    """Report-likeness score from report_classifier [0.0, 1.0]."""
    
    # Scores
    relevance_score: float = 0.0
    """Query-document relevance [0.0, 1.0]."""
    
    report_validity_score: float = 0.0
    """Report structure validity [0.0, 1.0]."""
    
    quality_score: float = 0.0
    """Analytical quality [0.0, 1.0]."""
    
    authority_score: float = 0.0
    """Source authority [0.0, 1.0]."""
    
    final_score: float = 0.0
    """Final ranking score [0.0, 1.0]. All results sorted by this descending."""
    
    score_breakdown: Optional[ScoreBreakdown] = None
    """All component scores in one object."""
    
    top_signals: Dict[str, Any] = field(default_factory=dict)
    """
    Top analytical signals extracted from the document.
    Example: {'methodology': 0.8, 'has_references': True, 'citation_score': 0.65}
    """
    
    reasoning: str = ""
    """Human-readable explanation of why this result ranked well."""
    
    warnings: List[str] = field(default_factory=list)
    """Warnings about trust/quality for this result."""
    
    index: int = 0
    """Rank position in final results."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.score_breakdown:
            data['score_breakdown'] = self.score_breakdown.to_dict()
        return data
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RankedReport":
        """Create instance from dictionary."""
        if 'score_breakdown' in data and isinstance(data['score_breakdown'], dict):
            data['score_breakdown'] = ScoreBreakdown.from_dict(data['score_breakdown'])
        return cls(**data)
    
    def quality_assessment(self) -> str:
        """Return a brief quality assessment based on scores."""
        avg = (
            self.relevance_score +
            self.report_validity_score +
            self.quality_score +
            self.authority_score
        ) / 4.0
        
        if avg >= 0.75:
            return "high"
        elif avg >= 0.50:
            return "medium"
        else:
            return "low"


@dataclass
class BatchResults:
    """Container for all ranked results for one query."""
    
    query: str
    """Original search query."""
    
    results: List[RankedReport] = field(default_factory=list)
    """List of ranked reports, sorted by final_score descending."""
    
    total_count: int = 0
    """Total number of results before top_k filtering."""
    
    returned_count: int = 0
    """Number of results actually returned (after filtering)."""
    
    iteration_count: int = 0
    """Number of iterations used to refine the query."""
    
    failure_type: Optional[str] = None
    """Last diagnosed failure type (if iteration was triggered)."""
    
    processing_time_ms: float = 0.0
    """How long the entire pipeline took (in milliseconds)."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query': self.query,
            'results': [r.to_dict() for r in self.results],
            'total_count': self.total_count,
            'returned_count': self.returned_count,
            'iteration_count': self.iteration_count,
            'failure_type': self.failure_type,
            'processing_time_ms': self.processing_time_ms,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchResults":
        """Create instance from dictionary."""
        results = [
            RankedReport.from_dict(r) for r in data.get('results', [])
        ]
        return cls(
            query=data['query'],
            results=results,
            total_count=data.get('total_count', 0),
            returned_count=data.get('returned_count', 0),
            iteration_count=data.get('iteration_count', 0),
            failure_type=data.get('failure_type'),
            processing_time_ms=data.get('processing_time_ms', 0.0),
        )
    
    def avg_score(self) -> float:
        """Average final score across all results."""
        if not self.results:
            return 0.0
        return sum(r.final_score for r in self.results) / len(self.results)
    
    def top_result(self) -> Optional[RankedReport]:
        """Return the top-ranked result, or None if empty."""
        return self.results[0] if self.results else None


def results_to_csv(results: List[RankedReport], filepath: str) -> None:
    """Export RankedReport objects to CSV."""
    import csv
    
    if not results:
        return
    
    fieldnames = [
        'index', 'title', 'url', 'year', 'source', 'source_class', 'report_type',
        'relevance_score', 'report_validity_score', 'quality_score', 'authority_score',
        'final_score', 'reasoning', 'warnings'
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {
                'index': result.index,
                'title': result.title,
                'url': result.url,
                'year': result.year or '',
                'source': result.source,
                'source_class': result.source_class,
                'report_type': result.report_type,
                'relevance_score': f"{result.relevance_score:.3f}",
                'report_validity_score': f"{result.report_validity_score:.3f}",
                'quality_score': f"{result.quality_score:.3f}",
                'authority_score': f"{result.authority_score:.3f}",
                'final_score': f"{result.final_score:.3f}",
                'reasoning': result.reasoning,
                'warnings': '; '.join(result.warnings) if result.warnings else '',
            }
            writer.writerow(row)


def batch_results_to_json(batch: BatchResults, filepath: str) -> None:
    """Export BatchResults to JSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(batch.to_json())


# ============================================================================
# TEST HARNESS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SCHEMAS TEST")
    print("=" * 80)
    
    # Create sample ScoreBreakdown
    scores = ScoreBreakdown(
        relevance_score=0.80,
        report_validity_score=0.72,
        quality_score=0.68,
        authority_score=0.85,
        final_score=0.75
    )
    print("\n1. ScoreBreakdown:")
    print(f"   Relevance: {scores.relevance_score}")
    print(f"   Validity: {scores.report_validity_score}")
    print(f"   Quality: {scores.quality_score}")
    print(f"   Authority: {scores.authority_score}")
    print(f"   Final: {scores.final_score}")
    print(f"   Average: {scores.avg_score():.3f}")
    
    # Create sample RankedReport
    report = RankedReport(
        query="machine learning market size",
        title="Global ML Market Analysis 2024",
        url="https://example.com/ml-market-2024",
        year=2024,
        source="mckinsey.com",
        source_class="consulting",
        authority_prior=0.75,
        report_type="report",
        relevance_score=0.80,
        report_validity_score=0.72,
        quality_score=0.68,
        authority_score=0.85,
        final_score=0.75,
        score_breakdown=scores,
        reasoning="High authority source, strong report structure, relevant to query",
        index=1,
    )
    print("\n2. RankedReport:")
    print(f"   Title: {report.title}")
    print(f"   Source: {report.source} ({report.source_class})")
    print(f"   Type: {report.report_type}")
    print(f"   Final Score: {report.final_score:.3f}")
    print(f"   Quality: {report.quality_assessment()}")
    
    # Serialization
    print("\n3. Serialization:")
    report_dict = report.to_dict()
    print(f"   to_dict() keys: {list(report_dict.keys())}")
    
    report_json = report.to_json(indent=2)
    print(f"   to_json():\n{report_json[:200]}...")
    
    # BatchResults
    batch = BatchResults(
        query="machine learning market size",
        results=[report],
        total_count=20,
        returned_count=1,
        iteration_count=1,
        failure_type=None,
        processing_time_ms=1250.5,
    )
    print("\n4. BatchResults:")
    print(f"   Query: {batch.query}")
    print(f"   Results: {batch.returned_count}/{batch.total_count}")
    print(f"   Avg Score: {batch.avg_score():.3f}")
    print(f"   Processing Time: {batch.processing_time_ms:.1f}ms")
    print(f"   Top Result: {batch.top_result().title if batch.top_result() else 'None'}")
    
    print("\n" + "=" * 80)
    print("✓ All schemas working correctly")
    print("=" * 80)
