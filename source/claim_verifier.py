"""Deterministic claim verification without an LLM.

This module mirrors the richer VerifiedClaim workflow in a lightweight form:
- claim objects can carry explicit, implicit, and cross-reference citations
- citations can be backtracked within nearby claim positions
- cited URLs are fetched with the existing document_fetcher
- claims are verified with lexical overlap and numeric matching

It intentionally avoids litellm, Pydantic, pandas, numpy, and prompt files.
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from source.classifier.source_classifier import classify_source
from source.fetching.document_fetcher import fetch_document
from source.verification_metrics import clean_url


ALLOWED_RESULTS = {"supported", "not_supported", "error"}
RESULT_PRIORITY = {
    "supported": 0,
    "not_supported": 1,
    "error": 2,
    "unverified": 3,
}

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "are", "was",
    "were", "has", "have", "had", "will", "can", "not", "but", "about", "among",
    "their", "its", "more", "than", "also", "such", "report", "study", "analysis",
}


@dataclass
class Claim:
    """One extracted claim with citation metadata."""

    position: str = ""
    claim: str = ""
    claim_type: str = "F"
    rationale: str = ""
    numeric: bool = False
    citations: list[str] = field(default_factory=list)
    implicit_citations: list[str] = field(default_factory=list)
    cross_references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    """Verification result for one claim against one URL/context."""

    claim: str
    explanation: str
    result: str
    url: str
    reliable_explanation: str = ""
    reliable: bool = False

    def __post_init__(self) -> None:
        if self.result == "conflict":
            self.result = "not_supported"
        if self.result not in ALLOWED_RESULTS:
            self.result = "not_supported"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerifiedClaim:
    """A claim plus all deterministic URL-level verification results."""

    claim: Claim
    verifications: list[VerificationResult] = field(default_factory=list)

    def final_result_and_explanation(self) -> tuple[str, str]:
        if not self.verifications:
            return "unverified", "This claim has not been verified."
        if len(self.verifications) == 1:
            return self.verifications[0].result, self.make_explanation()

        best_result = min(
            self.verifications,
            key=lambda item: RESULT_PRIORITY.get(item.result, RESULT_PRIORITY["not_supported"]),
        )
        return best_result.result, self.make_explanation()

    def make_explanation(self) -> str:
        lines: list[str] = []
        for verification in self.verifications:
            badge = make_badge(verification.result)
            lines.append(f"- {badge}: {verification.explanation} [Source]({verification.url})")
        return "\n" + "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "verifications": [verification.to_dict() for verification in self.verifications],
        }


_BADGE_LABELS = {
    "supported": "Supported",
    "not_supported": "Not supported",
    "error": "Error",
    "unverified": "Unverified",
}


def make_badge(badge_type: str, badge_text: str = "") -> str:
    """Return a plain text badge string suitable for CLI output."""
    normalized = "not_supported" if badge_type in {"partially_supported", "conflict"} else badge_type
    label = _BADGE_LABELS.get(normalized, normalized)
    suffix = f": {badge_text}" if badge_text else ""
    return f"[{label}{suffix}]"


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_PATTERN.findall(str(text or ""))
        if len(token) > 2 and token.lower() not in _STOPWORDS
    }


def _numbers(text: str) -> set[str]:
    return {item.replace(",", "") for item in _NUMBER_PATTERN.findall(str(text or ""))}


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    words = str(text or "").split()
    if not words:
        return []
    return [" ".join(words[index:index + chunk_size]) for index in range(0, len(words), chunk_size)]


def limit_context_by_overlap(
    queries: list[str],
    content: str,
    chunk_size: int = 1000,
    top_k: int = 2,
) -> str:
    """Select the most relevant chunks by token overlap with claim text."""
    chunks = _chunk_text(content, max(50, chunk_size))
    if not chunks:
        return ""

    query_tokens = set()
    for query in queries:
        query_tokens.update(_tokens(query))
    if not query_tokens:
        return "\n\n".join(chunks[:top_k])

    scored: list[tuple[int, int, str]] = []
    for index, chunk in enumerate(chunks):
        overlap = len(query_tokens & _tokens(chunk))
        scored.append((overlap, -index, chunk))

    scored.sort(reverse=True)
    selected = [chunk for _, _, chunk in scored[:max(1, top_k)]]
    return "\n\n".join(selected)


def _source_reliability(url: str, context: str = "") -> tuple[bool, str]:
    source_info = classify_source(url=url, text=context[:1000])
    authority_prior = float(source_info.get("authority_prior", 0.45) or 0.45)
    source_class = str(source_info.get("source_class", "unknown"))
    reliable = authority_prior >= 0.55 and source_class != "marketing_site"
    explanation = f"source_class={source_class}, authority_prior={authority_prior:.2f}"
    return reliable, explanation


def _verify_one_claim_against_context(claim: Claim, context: str, url: str) -> VerificationResult:
    reliable, reliable_explanation = _source_reliability(url, context)
    context_tokens = _tokens(context)
    claim_tokens = _tokens(claim.claim)

    if not context.strip():
        return VerificationResult(
            claim=claim.claim,
            result="error",
            explanation="No fetched context was available for this URL.",
            url=url,
            reliable=False,
            reliable_explanation="URL fetch failed or returned empty text.",
        )

    if not claim_tokens:
        return VerificationResult(
            claim=claim.claim,
            result="not_supported",
            explanation="Claim did not contain enough checkable terms.",
            url=url,
            reliable=reliable,
            reliable_explanation=reliable_explanation,
        )

    overlap = len(claim_tokens & context_tokens) / len(claim_tokens)
    claim_numbers = _numbers(claim.claim)
    context_numbers = _numbers(context)
    numbers_supported = bool(claim_numbers) and claim_numbers.issubset(context_numbers)

    if numbers_supported:
        result = "supported"
        explanation = "All numeric values in the claim appear in the cited context."
    elif overlap >= 0.60 and not claim_numbers:
        result = "supported"
        explanation = f"Claim terms substantially overlap with the cited context ({overlap:.0%})."
    elif overlap >= 0.75:
        result = "supported"
        explanation = f"Claim terms strongly overlap with the cited context ({overlap:.0%})."
    else:
        result = "not_supported"
        explanation = f"Only limited claim/context overlap was found ({overlap:.0%})."

    return VerificationResult(
        claim=claim.claim,
        result=result,
        explanation=explanation,
        url=url,
        reliable=reliable,
        reliable_explanation=reliable_explanation,
    )


async def fetch_webpages(urls: set[str]) -> dict[str, str]:
    """Fetch URL text concurrently using the existing synchronous fetcher."""
    async def fetch_one(url: str) -> tuple[str, str]:
        cleaned = clean_url(url)
        try:
            result = await asyncio.to_thread(fetch_document, cleaned)
        except Exception as exc:
            return cleaned, f"Failed to fetch: {exc}"
        if result.get("status") != "ok":
            return cleaned, f"Failed to fetch: {result.get('error', 'unknown_error')}"
        return cleaned, str(result.get("raw_text", ""))

    pairs = await asyncio.gather(*(fetch_one(url) for url in urls))
    return dict(pairs)


async def verify_claim_from_url_async(
    claims: list[Claim],
    context: str,
    url: str,
    limit_tokens: bool = True,
    top_k: int = 2,
    chunk_size: int = 1000,
    return_output: bool = False,
    **_: Any,
) -> list[VerificationResult] | tuple[list[VerificationResult], dict[str, Any]]:
    """Verify claims against one URL context without using an LLM."""
    if limit_tokens:
        context = limit_context_by_overlap(
            queries=[claim.claim for claim in claims],
            content=context,
            chunk_size=chunk_size,
            top_k=top_k,
        )

    results = [_verify_one_claim_against_context(claim, context, clean_url(url)) for claim in claims]
    if return_output:
        return results, {"method": "deterministic_overlap", "url": clean_url(url)}
    return results


async def batch_verify_claim_from_url_async(
    claims: list[Claim],
    context: str,
    url: str,
    top_k: int = 2,
    chunk_size: int = 1000,
    batch_size: int = 20,
    **kwargs: Any,
) -> list[VerificationResult]:
    """Verify claims in small batches against one URL context."""
    tasks = []
    for index in range(0, len(claims), batch_size):
        batch_claims = claims[index:index + batch_size]
        tasks.append(
            verify_claim_from_url_async(
                batch_claims,
                context=context,
                url=url,
                top_k=top_k,
                chunk_size=chunk_size,
                **kwargs,
            )
        )

    batches = await asyncio.gather(*tasks)
    results: list[VerificationResult] = []
    for batch in batches:
        if isinstance(batch, tuple):
            batch = batch[0]
        results.extend(batch)
    return results


def claim_by_position(claims: list[Claim]) -> dict[str, list[Claim]]:
    """Group claims by their position field."""
    position_to_claim: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        position_to_claim[claim.position].append(claim)
    return position_to_claim


def backtrack_claims(claims: list[Claim]) -> list[Claim]:
    """Backtrack missing citations from same-line neighbors and cross references."""
    line_to_claims: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        line_part = claim.position.split(".")[0] if claim.position else ""
        if line_part:
            line_to_claims[line_part].append(claim)

    for line_claims in line_to_claims.values():
        def sentence_index(item: Claim) -> int:
            match = re.search(r"\.S(\d+)", item.position)
            return int(match.group(1)) if match else 0

        line_claims.sort(key=sentence_index)
        active_citations: list[str] = []
        for claim in reversed(line_claims):
            if claim.citations:
                active_citations = list(claim.citations)
            if not claim.citations and not claim.implicit_citations and active_citations:
                claim.implicit_citations = list(active_citations)

    position_to_claim = claim_by_position(claims)
    for claim in claims:
        for cross_ref in claim.cross_references:
            for source_claim in position_to_claim.get(cross_ref, []):
                inherited = list(source_claim.citations) + list(source_claim.implicit_citations)
                if inherited:
                    claim.implicit_citations.extend(inherited)

    for claim in claims:
        if claim.claim_type in {"B", "C"} and not claim.citations and not claim.implicit_citations:
            claim.claim_type = "D"

    return claims


def replace_claims_to_url(claims: list[Claim], citations: dict[str, str]) -> list[Claim]:
    """Replace citation ids with URLs in-place and return claims."""
    for claim in claims:
        claim.citations = [citations.get(item, item) for item in claim.citations]
        claim.implicit_citations = [citations.get(item, item) for item in claim.implicit_citations]
    return claims


async def verify_claims_async(
    claims: list[Claim],
    return_context: bool = False,
    **kwargs: Any,
) -> list[VerifiedClaim] | tuple[list[VerifiedClaim], dict[str, dict[str, Any]]]:
    """Fetch cited URLs and verify claims deterministically."""
    backtracked_claims = backtrack_claims(claims)
    urls: set[str] = set()
    for claim in backtracked_claims:
        urls.update(clean_url(url) for url in claim.citations if clean_url(url))
        urls.update(clean_url(url) for url in claim.implicit_citations if clean_url(url))

    contexts = await fetch_webpages(urls)
    claim_to_verified = {
        claim.claim: VerifiedClaim(claim=claim, verifications=[])
        for claim in backtracked_claims
    }
    claims_by_url: dict[str, list[Claim]] = defaultdict(list)
    for claim in backtracked_claims:
        for url in set(list(claim.citations) + list(claim.implicit_citations)):
            cleaned = clean_url(url)
            if cleaned:
                claims_by_url[cleaned].append(claim)

    context_informations: dict[str, dict[str, Any]] = {}

    async def process_url(url: str) -> None:
        context = contexts.get(url, "")
        claim_list = claims_by_url[url]
        if not context or context.startswith("Failed to fetch"):
            for claim in claim_list:
                claim_to_verified[claim.claim].verifications.append(
                    VerificationResult(
                        claim=claim.claim,
                        result="error",
                        explanation="Failed to fetch content from URL.",
                        url=url,
                        reliable=False,
                        reliable_explanation="URL fetch failed or content was empty.",
                    )
                )
            context_informations[url] = {
                "context": context,
                "reliable": False,
                "reliable_explanation": "URL fetch failed or content was empty.",
            }
            return

        results = await batch_verify_claim_from_url_async(claim_list, context=context, url=url, **kwargs)
        for result in results:
            if result.claim in claim_to_verified:
                claim_to_verified[result.claim].verifications.append(result)

        first = results[0] if results else None
        context_informations[url] = {
            "context": context,
            "reliable": bool(first.reliable) if first else False,
            "reliable_explanation": first.reliable_explanation if first else "No verification results.",
        }

    await asyncio.gather(*(process_url(url) for url in claims_by_url))
    verified = list(claim_to_verified.values())
    if return_context:
        return verified, context_informations
    return verified
