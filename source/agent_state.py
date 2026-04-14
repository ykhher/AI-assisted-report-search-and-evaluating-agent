"""
The goal of this module is to give the agent a practical working memory.
It keeps track of what has been tried, which URLs were seen or failed, which
documents are still in play.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    """Return the current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CandidateRecord:
    """Track one candidate document as it moves through the agent loop."""

    url: str
    """Canonical URL used to identify the candidate."""

    title: str = ""
    """Candidate title when available from search results or fetched content."""

    source: str = ""
    """Publisher or source domain."""

    status: str = "discovered"
    """Current lifecycle state such as discovered, filtered_out, scored, or ranked."""

    score: float | None = None
    """Best known final score for this candidate, if scoring has happened."""

    notes: list[str] = field(default_factory=list)
    """Small debug notes explaining why the candidate was kept or rejected."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Extra candidate data kept in plain dict form for easy serialization."""

    def add_note(self, note: str) -> None:
        """Attach a short note when something important happens to this candidate."""
        cleaned = str(note).strip()
        if cleaned:
            self.notes.append(cleaned)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for logging or JSON serialization."""
        return asdict(self)

    def to_payload(self) -> dict[str, Any]:
        """Convert the candidate into the payload shape used by tools.
        """
        payload = dict(self.metadata)
        payload.setdefault("url", self.url)
        payload.setdefault("title", self.title)
        payload.setdefault("source", self.source)
        payload.setdefault("status", self.status)
        payload.setdefault("notes", list(self.notes))
        if self.score is not None:
            payload.setdefault("score", self.score)
        return payload


@dataclass
class ActionRecord:
    """Record one agent action for later inspection or debugging."""

    action: str
    """Action name such as search, filter_reports, or extract_signals."""

    status: str = "started"
    """Outcome status, typically started, completed, skipped, or failed."""

    detail: str = ""
    """Human-readable detail explaining what the action attempted."""

    timestamp: str = field(default_factory=_utc_now_iso)
    """UTC timestamp captured when the action record was created."""

    payload: dict[str, Any] = field(default_factory=dict)
    """Optional lightweight payload with relevant context."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for logging or JSON serialization."""
        return asdict(self)


@dataclass
class AgentState:
    """Working state for a single report discovery and ranking run."""

    user_query: str
    """Original raw request from the user."""

    current_plan: dict[str, Any] = field(default_factory=dict)
    """Planner output describing task type, priorities, and suggested steps."""

    rewritten_queries_tried: list[str] = field(default_factory=list)
    """All refined or expanded queries attempted during the run."""

    visited_urls: set[str] = field(default_factory=set)
    """URLs already seen so the agent can avoid duplicate work."""

    failed_urls: dict[str, str] = field(default_factory=dict)
    """URL -> failure reason for fetch, parse, or trust-related failures."""

    candidates: list[CandidateRecord] = field(default_factory=list)
    """Documents still under consideration or already scored."""

    filtered_out: list[CandidateRecord] = field(default_factory=list)
    """Documents rejected from the candidate set, with notes when possible."""

    reflection_notes: list[str] = field(default_factory=list)
    """Short reasoning notes about quality, failure modes, and next actions."""

    failure_history: list[str] = field(default_factory=list)
    """Chronological list of diagnosed failure types across iterations."""

    action_history: list[ActionRecord] = field(default_factory=list)
    """Compact trace of actions taken by the agent."""

    current_step: str = "plan"
    """Current step in the execution loop."""

    stop_reason: str | None = None
    """Why the agent stopped, such as sufficient_quality or no_results."""

    started_at: str = field(default_factory=_utc_now_iso)
    """UTC start time for the run."""

    processing_time_ms: float = 0.0
    """Elapsed processing time in milliseconds when the run completes."""

    def add_candidate(
        self,
        url: str,
        title: str = "",
        source: str = "",
        score: float | None = None,
        status: str = "discovered",
        metadata: dict[str, Any] | None = None,
    ) -> CandidateRecord:
        """Add a candidate if it has not already been seen.

        Returning the record makes it easy for the caller to attach notes or
        update status without looking it up again.
        """
        normalized_url = str(url).strip()
        if not normalized_url:
            raise ValueError("Candidate URL cannot be empty.")

        existing = self.get_candidate(normalized_url)
        if existing is not None:
            if title and not existing.title:
                existing.title = title
            if source and not existing.source:
                existing.source = source
            if score is not None:
                existing.score = score
            if metadata:
                existing.metadata.update(metadata)
            return existing

        candidate = CandidateRecord(
            url=normalized_url,
            title=str(title or ""),
            source=str(source or ""),
            status=str(status or "discovered"),
            score=score,
            metadata=dict(metadata or {}),
        )
        self.candidates.append(candidate)
        self.visited_urls.add(normalized_url)
        return candidate

    def get_candidate(self, url: str) -> CandidateRecord | None:
        """Return a candidate by URL if it is already tracked."""
        normalized_url = str(url).strip()
        for candidate in self.candidates:
            if candidate.url == normalized_url:
                return candidate
        return None

    def filter_out_candidate(self, url: str, reason: str) -> None:
        """Move a candidate out of the active set and preserve the rejection note."""
        normalized_url = str(url).strip()
        if not normalized_url:
            return

        kept: list[CandidateRecord] = []
        moved = False
        for candidate in self.candidates:
            if candidate.url == normalized_url and not moved:
                candidate.status = "filtered_out"
                candidate.add_note(reason)
                self.filtered_out.append(candidate)
                moved = True
            else:
                kept.append(candidate)
        self.candidates = kept

    def mark_url_failed(self, url: str, reason: str) -> None:
        """Record that a URL failed and keep the reason for later reflection."""
        normalized_url = str(url).strip()
        cleaned_reason = str(reason).strip() or "unknown_failure"
        if not normalized_url:
            return

        self.failed_urls[normalized_url] = cleaned_reason
        self.visited_urls.add(normalized_url)

        candidate = self.get_candidate(normalized_url)
        if candidate is not None:
            candidate.status = "failed"
            candidate.add_note(cleaned_reason)

    def add_reflection(self, note: str) -> None:
        """Add a concise reasoning note about the current state of the run."""
        cleaned = str(note).strip()
        if cleaned:
            self.reflection_notes.append(cleaned)

    def add_failure(self, failure_type: str) -> None:
        """Append a diagnosed failure type if one is available."""
        cleaned = str(failure_type).strip()
        if cleaned:
            self.failure_history.append(cleaned)

    def record_action(
        self,
        action: str,
        status: str = "completed",
        detail: str = "",
        payload: dict[str, Any] | None = None,
    ) -> ActionRecord:
        """Append a compact action record to the execution trace."""
        record = ActionRecord(
            action=str(action).strip(),
            status=str(status).strip() or "completed",
            detail=str(detail).strip(),
            payload=dict(payload or {}),
        )
        self.action_history.append(record)
        return record

    def mark_stopped(self, reason: str, processing_time_ms: float | None = None) -> None:
        """Set the stop reason and optionally capture elapsed runtime."""
        self.stop_reason = str(reason).strip() or "stopped"
        if processing_time_ms is not None:
            self.processing_time_ms = float(processing_time_ms)

    def to_dict(self) -> dict[str, Any]:
        """Convert the full state into a serialization-friendly dictionary."""
        return {
            "user_query": self.user_query,
            "current_plan": dict(self.current_plan),
            "rewritten_queries_tried": list(self.rewritten_queries_tried),
            "visited_urls": sorted(self.visited_urls),
            "failed_urls": dict(self.failed_urls),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "filtered_out": [candidate.to_dict() for candidate in self.filtered_out],
            "reflection_notes": list(self.reflection_notes),
            "failure_history": list(self.failure_history),
            "action_history": [record.to_dict() for record in self.action_history],
            "current_step": self.current_step,
            "stop_reason": self.stop_reason,
            "started_at": self.started_at,
            "processing_time_ms": self.processing_time_ms,
        }

    def active_candidate_payloads(self) -> list[dict[str, Any]]:
        """Return active candidate payload dictionaries for tools and reflection."""
        return [candidate.to_payload() for candidate in self.candidates]
