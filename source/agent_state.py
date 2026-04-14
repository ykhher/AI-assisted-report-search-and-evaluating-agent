"""State objects for one report discovery run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    """Return the current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CandidateRecord:
    """One candidate document tracked by the agent."""

    url: str
    title: str = ""
    source: str = ""
    status: str = "discovered"
    score: float | None = None
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_note(self, note: str) -> None:
        """Store a short note about this candidate."""
        note = str(note).strip()
        if note:
            self.notes.append(note)

    def to_payload(self) -> dict[str, Any]:
        """Merge the fixed fields with metadata for tool calls."""
        payload = dict(self.metadata)
        payload.setdefault("url", self.url)
        payload.setdefault("title", self.title)
        payload.setdefault("source", self.source)
        payload.setdefault("status", self.status)
        payload.setdefault("notes", list(self.notes))
        if self.score is not None:
            payload.setdefault("score", self.score)
        return payload

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionRecord:
    """A small trace item for one controller action."""

    action: str
    status: str = "completed"
    detail: str = ""
    timestamp: str = field(default_factory=_utc_now_iso)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    """Working memory for a single run."""

    user_query: str
    current_plan: dict[str, Any] = field(default_factory=dict)
    rewritten_queries_tried: list[str] = field(default_factory=list)
    visited_urls: set[str] = field(default_factory=set)
    failed_urls: dict[str, str] = field(default_factory=dict)
    candidates: list[CandidateRecord] = field(default_factory=list)
    filtered_out: list[CandidateRecord] = field(default_factory=list)
    reflection_notes: list[str] = field(default_factory=list)
    failure_history: list[str] = field(default_factory=list)
    action_history: list[ActionRecord] = field(default_factory=list)
    current_step: str = "plan"
    stop_reason: str | None = None
    started_at: str = field(default_factory=_utc_now_iso)
    processing_time_ms: float = 0.0

    def get_candidate(self, url: str) -> CandidateRecord | None:
        """Find one candidate by URL."""
        url = str(url).strip()
        for candidate in self.candidates:
            if candidate.url == url:
                return candidate
        return None

    def add_candidate(
        self,
        url: str,
        title: str = "",
        source: str = "",
        score: float | None = None,
        status: str = "discovered",
        metadata: dict[str, Any] | None = None,
    ) -> CandidateRecord:
        """Add a new candidate, or update an existing one."""
        url = str(url).strip()
        if not url:
            raise ValueError("Candidate URL cannot be empty.")

        existing = self.get_candidate(url)
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
            url=url,
            title=str(title or ""),
            source=str(source or ""),
            status=str(status or "discovered"),
            score=score,
            metadata=dict(metadata or {}),
        )
        self.candidates.append(candidate)
        self.visited_urls.add(url)
        return candidate

    def filter_out_candidate(self, url: str, reason: str) -> None:
        """Move a candidate from active results into the filtered list."""
        candidate = self.get_candidate(url)
        if candidate is None:
            return

        candidate.status = "filtered_out"
        candidate.add_note(reason)
        self.filtered_out.append(candidate)
        self.candidates = [item for item in self.candidates if item.url != candidate.url]

    def mark_url_failed(self, url: str, reason: str) -> None:
        """Remember that this URL failed so we do not retry blindly."""
        url = str(url).strip()
        reason = str(reason).strip() or "unknown_failure"
        if not url:
            return

        self.failed_urls[url] = reason
        self.visited_urls.add(url)

        candidate = self.get_candidate(url)
        if candidate is not None:
            candidate.status = "failed"
            candidate.add_note(reason)

    def add_reflection(self, note: str) -> None:
        note = str(note).strip()
        if note:
            self.reflection_notes.append(note)

    def add_failure(self, failure_type: str) -> None:
        failure_type = str(failure_type).strip()
        if failure_type:
            self.failure_history.append(failure_type)

    def record_action(
        self,
        action: str,
        status: str = "completed",
        detail: str = "",
        payload: dict[str, Any] | None = None,
    ) -> ActionRecord:
        """Store a compact action log entry."""
        record = ActionRecord(
            action=str(action).strip(),
            status=str(status).strip() or "completed",
            detail=str(detail).strip(),
            payload=dict(payload or {}),
        )
        self.action_history.append(record)
        return record

    def mark_stopped(self, reason: str, processing_time_ms: float | None = None) -> None:
        self.stop_reason = str(reason).strip() or "stopped"
        if processing_time_ms is not None:
            self.processing_time_ms = float(processing_time_ms)

    def active_candidate_payloads(self) -> list[dict[str, Any]]:
        """This is the shape most tools expect."""
        return [candidate.to_payload() for candidate in self.candidates]

    def to_dict(self) -> dict[str, Any]:
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
