"""Optional local Qwen integration for query rewriting and lightweight agent assistance."""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


def _unique_paths(paths: list[Path]) -> list[Path]:
    """Return existing paths while preserving order and removing duplicates."""
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            resolved = str(path.expanduser().resolve())
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if Path(resolved).exists():
            unique.append(Path(resolved))
    return unique


def _candidate_site_packages() -> list[Path]:
    """Look for site-packages folders that may already contain Transformers/Torch."""
    home = Path.home()
    candidates: list[Path] = []

    extra_site_packages = os.environ.get("QWEN_EXTRA_SITE_PACKAGES")
    if extra_site_packages:
        candidates.append(Path(extra_site_packages))

    desktop_roots = [home / "Desktop", home / "OneDrive" / "Desktop"]
    for root in desktop_roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            site_packages = child / ".venv" / "Lib" / "site-packages"
            if site_packages.exists():
                candidates.append(site_packages)

    return _unique_paths(candidates)


def _candidate_model_paths() -> list[Path]:
    """Discover local Qwen model directories, preferring smaller instruct models first."""
    home = Path.home()
    candidates: list[Path] = []

    explicit = os.environ.get("QWEN_MODEL_PATH")
    if explicit:
        candidates.append(Path(explicit))

    hub_root = home / ".cache" / "huggingface" / "hub"
    model_names = [
        "models--Qwen--Qwen2.5-0.5B-Instruct",
        "models--Qwen--Qwen2.5-1.5B-Instruct",
        "models--Qwen--Qwen2.5-3B-Instruct",
        "models--Qwen--Qwen2.5-7B-Instruct",
        "models--Qwen--Qwen2.5-7B-Instruct-1M",
    ]
    for model_name in model_names:
        candidates.append(hub_root / model_name)

    desktop_roots = [home / "Desktop", home / "OneDrive" / "Desktop"]
    for root in desktop_roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and "qwen" in child.name.lower():
                candidates.append(child)
            for nested in child.glob("**/Qwen*"):
                if nested.is_dir():
                    candidates.append(nested)

    discovered: list[Path] = []
    for candidate in _unique_paths(candidates):
        if (candidate / "config.json").exists():
            discovered.append(candidate)
            continue

        snapshots_dir = candidate / "snapshots"
        if snapshots_dir.exists():
            for snapshot in sorted(snapshots_dir.iterdir()):
                if (snapshot / "config.json").exists():
                    discovered.append(snapshot)

    return _unique_paths(discovered)


@lru_cache(maxsize=1)
def _ensure_ml_stack() -> tuple[Any, Any, Any] | None:
    """Import Torch/Transformers, reusing other local virtualenvs when needed."""
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        return torch, AutoModelForCausalLM, AutoTokenizer
    except Exception:
        for site_packages in _candidate_site_packages():
            site_packages_str = str(site_packages)
            if site_packages_str not in sys.path:
                sys.path.append(site_packages_str)
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            return torch, AutoModelForCausalLM, AutoTokenizer
        except Exception:
            return None


def get_local_qwen_status() -> dict[str, Any]:
    """Return whether local Qwen support is discoverable on this machine."""
    model_path = next(iter(_candidate_model_paths()), None)
    ml_stack = _ensure_ml_stack()
    return {
        "available": bool(model_path and ml_stack),
        "model_path": str(model_path) if model_path else None,
        "borrowed_site_packages": [str(path) for path in _candidate_site_packages()],
    }


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any, Any] | None:
    """Load the local Qwen model lazily and keep it cached for reuse."""
    ml_stack = _ensure_ml_stack()
    model_path = next(iter(_candidate_model_paths()), None)
    if not ml_stack or model_path is None:
        return None

    torch, AutoModelForCausalLM, AutoTokenizer = ml_stack
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.float32,
    )
    model.eval()
    return tokenizer, model, torch


def _generate(
    prompt: str,
    max_new_tokens: int = 120,
    system_prompt: str | None = None,
) -> str:
    """Generate a small completion using the locally cached Qwen model."""
    loaded = _load_model()
    if loaded is None:
        return ""

    tokenizer, model, torch = loaded
    messages = [
        {
            "role": "system",
            "content": system_prompt or (
                "You improve web-search queries for market and industry report discovery. "
                "Return only a JSON array of concise search queries."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        rendered_prompt = (
            "Return only a JSON array of concise search queries.\n"
            f"User request: {prompt}\nAssistant:"
        )

    inputs = tokenizer(rendered_prompt, return_tensors="pt", truncation=True, max_length=2048)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _clean_query(text: str) -> str:
    """Normalize a generated query into a compact search string."""
    cleaned = str(text).strip()
    dict_match = re.search(r"['\"]?(?:query|search_query|text)['\"]?\s*:\s*['\"]([^'\"]+)['\"]", cleaned)
    if dict_match:
        cleaned = dict_match.group(1)

    cleaned = re.sub(r"^[\-\d\.]\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" {}\"'")


def _normalize_query_item(item: Any) -> str:
    """Extract a query string from either a raw string or a dictionary-like payload."""
    if isinstance(item, dict):
        for key in ("query", "search_query", "text"):
            value = item.get(key)
            if value:
                return _clean_query(value)
    return _clean_query(str(item))


def _parse_generated_queries(raw_output: str) -> list[str]:
    """Parse a JSON list or newline-separated model output into search queries."""
    if not raw_output:
        return []

    match = re.search(r"\[[\s\S]*\]", raw_output)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [_normalize_query_item(item) for item in parsed if _normalize_query_item(item)]
        except json.JSONDecodeError:
            pass

    queries: list[str] = []
    for line in raw_output.splitlines():
        cleaned = _normalize_query_item(line)
        if cleaned:
            queries.append(cleaned)
    return queries


def _clamp01(value: Any) -> float:
    """Clamp an arbitrary numeric value into the inclusive range [0, 1]."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(numeric, 1.0))


def _parse_json_object(raw_output: str) -> dict[str, Any]:
    """Extract a JSON-like object from model output, tolerating minor formatting drift."""
    if not raw_output:
        return {}

    match = re.search(r"\{[\s\S]*\}", raw_output)
    if not match:
        return {}

    snippet = match.group(0)
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {}


def _extract_named_score(raw_output: str, *names: str) -> float:
    """Fallback regex extraction for score labels when strict JSON parsing fails."""
    for name in names:
        patterns = [name, name.replace("_", " ")]
        for pattern_name in patterns:
            match = re.search(
                rf"{re.escape(pattern_name)}['\"]?\s*[:=]\s*([01](?:\.\d+)?)",
                raw_output,
                re.IGNORECASE,
            )
            if match:
                return _clamp01(match.group(1))
    return 0.0


def _prepare_signal_excerpt(text: str, max_chars: int = 2600) -> str:
    """Keep the beginning and end of long text so methods and footnotes both remain visible."""
    raw_text = str(text or "").strip()
    if len(raw_text) <= max_chars:
        return raw_text
    head = raw_text[:1600].rstrip()
    tail = raw_text[-900:].lstrip()
    return f"{head}\n...\n{tail}"


@lru_cache(maxsize=128)
def assess_text_signals(text: str, source: str = "") -> dict[str, Any]:
    """Use the local Qwen model to score references, methodology, consistency, and source authority."""
    default = {
        "reference_score": 0.0,
        "methodology_score": 0.0,
        "consistency_score": 0.0,
        "source_score": 0.0,
        "reason": "",
    }

    if os.environ.get("USE_LOCAL_QWEN_SIGNALS", "1").strip().lower() in {"0", "false", "no"}:
        return default

    excerpt = _prepare_signal_excerpt(text)
    source_text = " ".join(str(source or "").strip().split())
    if not excerpt and not source_text:
        return default

    system_prompt = (
        "You are a strict credibility-scoring function for business and consulting reports. "
        "Consulting PDFs often cite sources with footnotes like '1 Source: IEA report 2024' instead of a References heading. "
        "Treat those as strong evidence of references. Also score the publisher or organization authority when a source name is shown. "
        "Return exactly one JSON object and nothing else. All scores must be floats between 0 and 1."
    )
    prompt = (
        "Score the report excerpt using this schema: "
        '{"reference_score": 0.82, "methodology_score": 0.74, "consistency_score": 0.68, "source_score": 0.85, "reason": "brief reason"}.\n'
        "Rules:\n"
        "- reference_score: citations, source notes, numbered references, institutional attributions, footnotes.\n"
        "- methodology_score: research design, survey description, benchmark process, sample, data collection.\n"
        "- consistency_score: whether claims, numbers, and conclusions are coherent and report-like rather than marketing fluff.\n"
        "- source_score: credibility of the publisher or named institution (e.g. World Bank, OECD, IEA, McKinsey, Oxford Economics).\n\n"
        "Positive example:\n"
        "Source: World Bank\n"
        "Excerpt: Results improved.1 We conducted a survey of 120 firms. 1 Source: IEA report 2024 2 Source: IPCC analysis\n"
        'Output: {"reference_score": 0.9, "methodology_score": 0.8, "consistency_score": 0.7, "source_score": 0.95, "reason": "survey methods and authoritative numbered sources"}\n\n'
        "Negative example:\n"
        "Source: unknown blog\n"
        "Excerpt: This blog shares opinions and trends with no data or sources.\n"
        'Output: {"reference_score": 0.05, "methodology_score": 0.05, "consistency_score": 0.1, "source_score": 0.15, "reason": "opinion only"}\n\n'
        "Now score the next excerpt and return JSON only.\n\n"
        f"Source: {source_text or 'unknown'}\n"
        f"Excerpt:\n{excerpt or 'No excerpt provided.'}"
    )

    raw_output = _generate(prompt, max_new_tokens=220, system_prompt=system_prompt)
    parsed = _parse_json_object(raw_output)

    reference_score = _clamp01(
        parsed.get("reference_score", parsed.get("citation_score", _extract_named_score(raw_output, "reference_score", "citation_score")))
    )
    methodology_score = _clamp01(
        parsed.get("methodology_score", parsed.get("methodology", _extract_named_score(raw_output, "methodology_score", "methodology")))
    )
    consistency_score = _clamp01(
        parsed.get("consistency_score", parsed.get("consistency", _extract_named_score(raw_output, "consistency_score", "consistency")))
    )
    source_score = _clamp01(
        parsed.get("source_score", parsed.get("source_reputation", _extract_named_score(raw_output, "source_score", "source_reputation")))
    )
    reason = str(parsed.get("reason") or "").strip()

    return {
        "reference_score": reference_score,
        "methodology_score": methodology_score,
        "consistency_score": consistency_score,
        "source_score": source_score,
        "reason": reason,
    }


def suggest_search_queries(user_input: str, max_queries: int = 3) -> list[str]:
    """Generate a few better search queries using the local Qwen model when available."""
    topic = " ".join(str(user_input or "").strip().split())
    if not topic:
        return []

    prompt = (
        f"Create {max_queries} concise web-search queries for finding credible market or industry reports about: {topic}. "
        "Each query should target PDFs, outlooks, benchmarks, forecasts, or research reports. "
        "Return strict JSON only."
    )
    raw_output = _generate(prompt)
    generated = _parse_generated_queries(raw_output)

    unique_queries: list[str] = []
    seen: set[str] = set()
    for query in generated:
        normalized = query.lower()
        if normalized not in seen:
            unique_queries.append(query)
            seen.add(normalized)
        if len(unique_queries) >= max_queries:
            break

    return unique_queries


def rewrite_search_query(user_input: str) -> str:
    """Return a single locally rewritten query, falling back to the original text."""
    suggestions = suggest_search_queries(user_input, max_queries=1)
    return suggestions[0] if suggestions else str(user_input).strip()
