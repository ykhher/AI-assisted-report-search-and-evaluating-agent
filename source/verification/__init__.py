"""Claim extraction, citation parsing, and lightweight post-ranking verification."""

from source.verification.citations import (
    Citation,
    assign_citations_to_claims,
    citation_debug_payload,
    extract_citation_records,
    extract_citations,
)
from source.verification.claims import (
    Claim,
    VerificationResult,
    VerifiedClaim,
    backtrack_claims,
    make_badge,
    verify_claims_async,
)
from source.verification.core import (
    attach_verification_notes,
    extract_key_claims,
)
from source.verification.metrics import (
    clean_url,
    compute_metrics,
    compute_metrics_from_notes,
)

__all__ = [
    "Citation",
    "Claim",
    "VerificationResult",
    "VerifiedClaim",
    "assign_citations_to_claims",
    "attach_verification_notes",
    "backtrack_claims",
    "citation_debug_payload",
    "clean_url",
    "compute_metrics",
    "compute_metrics_from_notes",
    "extract_citation_records",
    "extract_citations",
    "extract_key_claims",
    "make_badge",
    "verify_claims_async",
]
