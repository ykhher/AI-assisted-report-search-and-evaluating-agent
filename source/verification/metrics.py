"""Verification metrics for lightweight claim and citation checks.

This module adapts the richer VerifiedClaim-style metric idea to this project.
It has no pandas/numpy dependency and also works with the current
`verification_notes` dictionaries produced by source.verification.
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any
from urllib.parse import urlsplit, urlunsplit


EXTERNAL_CLAIM_TYPES = {"A", "B", "C"}
CONTENT_CLAIM_TYPES = {"A", "B", "C", "D", "E"}


def clean_url(url: Any) -> str:
    """Normalize a URL for reference counting."""
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw.lower()

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, "", ""))


def _read(value: Any, key: str, default: Any = None) -> Any:
    """Read a field from a dict or object."""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _read_nested(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for part in path:
        current = _read(current, part, default)
        if current is default:
            return default
    return current


def _final_result(claim: Any) -> str:
    """Return the verification result for a claim-like object."""
    if isinstance(claim, dict) and claim.get("result"):
        return str(claim["result"])

    method = getattr(claim, "final_result_and_explanation", None)
    if callable(method):
        try:
            result, _ = method()
            return str(result)
        except Exception:
            return "error"

    return str(_read(claim, "result", "unsupported"))


def _claim_type(claim: Any) -> str:
    return str(_read_nested(claim, ["claim", "claim_type"], _read(claim, "claim_type", "F")) or "F")


def _is_numeric(claim: Any) -> bool:
    return bool(_read_nested(claim, ["claim", "numeric"], _read(claim, "numeric", False)))


def _verifications(claim: Any) -> list[Any]:
    value = _read(claim, "verifications", [])
    return value if isinstance(value, list) else []


def compute_claim_metrics(verified_claims: list[Any]) -> dict[str, Any]:
    """Compute claim-level metrics and compact claim rows."""
    claim_type_counts: dict[str, int] = {}
    claims: list[dict[str, Any]] = []

    for claim in verified_claims:
        claim_type = _claim_type(claim)
        result = _final_result(claim)
        claim_type_counts[claim_type] = claim_type_counts.get(claim_type, 0) + 1

        claims.append(
            {
                "claim_type": claim_type,
                "result": result,
                "supported": result == "supported",
                "error": result == "error",
                "is_numeric": _is_numeric(claim),
                "is_common": claim_type == "E",
                "is_internal": claim_type == "D",
                "is_external": claim_type in {"A", "B", "C", "F"},
            }
        )

    return {
        "total_claims": len(verified_claims),
        "claim_type_counts": claim_type_counts,
        "claims": claims,
    }


def compute_citation_metrics(verified_claims: list[Any]) -> dict[str, int]:
    """Compute citation support counts for externally cited claims."""
    num_citations = 0
    num_supported = 0
    num_error_citations = 0

    for claim in verified_claims:
        if _claim_type(claim) not in EXTERNAL_CLAIM_TYPES:
            continue

        for verification in _verifications(claim):
            result = str(_read(verification, "result", "unsupported"))
            num_citations += 1
            if result == "error":
                num_error_citations += 1
            if result == "supported":
                num_supported += 1

    return {
        "num_supported": num_supported,
        "num_error_citations": num_error_citations,
        "num_citations": num_citations,
    }


def _hhi_from_counts(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    shares = [count / total for count in counts]
    return sum(share * share for share in shares)


def _hhi_balance_score_0_10(counts: list[int]) -> float:
    """Return 0-10 citation balance score; 10 means supported citations are evenly spread."""
    positive_counts = [count for count in counts if count > 0]
    n = len(positive_counts)
    if n <= 1:
        return 0.0

    hhi = _hhi_from_counts(positive_counts)
    hhi_norm = (hhi - 1 / n) / (1 - 1 / n)
    return max(0.0, min(10.0, 10 * (1 - hhi_norm)))


def compute_reference_metrics(
    verified_claims: list[Any],
    references: dict[str, Any] | None = None,
    context_informations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute reference usage, support, reliability, and diversity metrics."""
    references = references or {}
    context_informations = context_informations or {}

    cited_counts: dict[str, int] = {}
    supported_cited_counts: dict[str, int] = {}
    supported_references: set[str] = set()
    used_references: set[str] = set()
    error_references: set[str] = set()
    reliable_references: set[str] = set()

    unique_references = {
        clean_url(value.get("url") if isinstance(value, dict) else value)
        for value in references.values()
        if clean_url(value.get("url") if isinstance(value, dict) else value)
    }

    for claim in verified_claims:
        for verification in _verifications(claim):
            url = clean_url(_read(verification, "url", ""))
            if not url:
                continue

            result = str(_read(verification, "result", "unsupported"))
            reliable = bool(_read(verification, "reliable", False))

            cited_counts[url] = cited_counts.get(url, 0) + 1
            used_references.add(url)

            if result == "supported":
                supported_references.add(url)
                supported_cited_counts[url] = supported_cited_counts.get(url, 0) + 1
            if result == "error":
                error_references.add(url)
            if reliable:
                reliable_references.add(url)

    for url in context_informations:
        cleaned = clean_url(url)
        if cleaned:
            unique_references.add(cleaned)

    all_counts = list(cited_counts.values())
    citations_mean = mean(all_counts) if all_counts else 0.0
    citations_std = pstdev(all_counts) if len(all_counts) > 1 else 0.0

    return {
        "num_references": len(references),
        "num_unique_references": len(unique_references),
        "num_supported": len(supported_references),
        "num_used": len(used_references),
        "num_error": len(error_references),
        "num_reliable": len(reliable_references & supported_references),
        "supported_references": sorted(supported_references),
        "used_references": sorted(used_references),
        "error_references": sorted(error_references),
        "cited_counts": cited_counts,
        "citations_mean": float(citations_mean),
        "citations_std": float(citations_std),
        "supported_cited_counts": supported_cited_counts,
        "balance_supported_0_10": float(_hhi_balance_score_0_10(list(supported_cited_counts.values()))),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def compute_metrics(
    verified_claims: list[Any],
    references: dict[str, Any] | None = None,
    context_informations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute integrity and sufficiency metrics from verified claims."""
    references = references or {}
    context_informations = {
        url: {key: value for key, value in info.items() if key != "context"}
        for url, info in (context_informations or {}).items()
        if isinstance(info, dict)
    }

    claim_metrics = compute_claim_metrics(verified_claims)
    citation_metrics = compute_citation_metrics(verified_claims)
    reference_metrics = compute_reference_metrics(verified_claims, references, context_informations)
    claims = claim_metrics["claims"]

    content_claims = [claim for claim in claims if claim["claim_type"] in CONTENT_CLAIM_TYPES]
    accurate_claims = [
        claim
        for claim in content_claims
        if claim["supported"] or claim["is_internal"] or claim["is_common"]
    ]
    external_claims = [claim for claim in claims if claim["claim_type"] in EXTERNAL_CLAIM_TYPES]
    external_numeric_claims = [claim for claim in external_claims if claim["is_numeric"]]

    claim_accuracy = _safe_ratio(len(accurate_claims), len(content_claims))
    external_claim_accuracy = _safe_ratio(sum(1 for claim in external_claims if claim["supported"]), len(external_claims))
    external_numeric_claim_accuracy = _safe_ratio(
        sum(1 for claim in external_numeric_claims if claim["supported"]),
        len(external_numeric_claims),
    )
    citation_accuracy = _safe_ratio(citation_metrics["num_supported"], citation_metrics["num_citations"])

    supported_per_shown = _safe_ratio(reference_metrics["num_supported"], reference_metrics["num_unique_references"])
    supported_per_used = _safe_ratio(reference_metrics["num_supported"], reference_metrics["num_used"])
    used_per_shown = _safe_ratio(reference_metrics["num_used"], reference_metrics["num_unique_references"])
    reproducibility_ratio = 1 - _safe_ratio(reference_metrics["num_error"], reference_metrics["num_used"])
    reliability_ratio = _safe_ratio(reference_metrics["num_reliable"], reference_metrics["num_used"])

    claim_types = claim_metrics["claim_type_counts"]
    num_verifiable_claims = sum(claim_types.get(key, 0) for key in ["A", "B", "C", "D"])
    num_externally_verifiable_claims = claim_types.get("A", 0) + claim_types.get("B", 0)
    num_unverifiable_claims = claim_types.get("F", 0)

    verified_claims_ratio = _safe_ratio(num_verifiable_claims, num_verifiable_claims + num_unverifiable_claims)
    externally_verifiable_claims_ratio = _safe_ratio(
        num_externally_verifiable_claims,
        num_externally_verifiable_claims + num_unverifiable_claims,
    )
    average_citations_per_claim = _safe_ratio(citation_metrics["num_citations"], num_externally_verifiable_claims)
    unique_reference = _safe_ratio(reference_metrics["num_supported"], citation_metrics["num_supported"])
    citations_cv = _safe_ratio(reference_metrics["citations_std"], reference_metrics["citations_mean"])

    return {
        "raw": {
            "claim_metrics": claim_metrics,
            "citation_metrics": citation_metrics,
            "reference_metrics": reference_metrics,
        },
        "integrity": {
            "claim_factuality": {
                "claim_accuracy": claim_accuracy,
                "external_claim_accuracy": external_claim_accuracy,
                "external_numeric_claim_accuracy": external_numeric_claim_accuracy,
            },
            "citation_validity": {
                "citation_accuracy": citation_accuracy,
            },
            "reference_accuracy": {
                "supported_per_shown": supported_per_shown,
                "supported_per_used": supported_per_used,
                "used_per_shown": used_per_shown,
            },
            "reference_quality": {
                "reproducibility": reproducibility_ratio,
                "reliability": reliability_ratio,
            },
            "reference_diversity": {
                "unique_reference": unique_reference,
                "citations_CV": citations_cv,
                "diversity_hhi": reference_metrics["balance_supported_0_10"],
            },
        },
        "sufficiency": {
            "source_support": {
                "verified_claims_ratio": verified_claims_ratio,
                "externally_verifiable_claims_ratio": externally_verifiable_claims_ratio,
                "average_citations_per_claim": average_citations_per_claim,
            },
            "information": len(accurate_claims),
            "citations": citation_metrics["num_supported"],
            "references": reference_metrics["num_supported"],
        },
    }


def compute_metrics_from_notes(
    verification_notes: list[dict[str, Any]],
    report_url: str = "",
) -> dict[str, Any]:
    """Compute metrics from this project's lightweight verification note format."""
    reference_url = clean_url(report_url)
    claims: list[dict[str, Any]] = []
    for note in verification_notes:
        verifications = note.get("verifications")
        if not isinstance(verifications, list):
            verifications = []
            if reference_url and note.get("result") in {"supported", "error"}:
                verifications.append(
                    {
                        "url": reference_url,
                        "result": note.get("result"),
                        "reliable": note.get("confidence") in {"medium", "high"},
                    }
                )

        claims.append(
            {
                "claim_type": note.get("claim_type", "F"),
                "result": note.get("result", "unsupported"),
                "numeric": bool(note.get("numeric", False)),
                "verifications": verifications,
            }
        )

    references = {"report": reference_url} if reference_url else {}
    return compute_metrics(claims, references=references)
