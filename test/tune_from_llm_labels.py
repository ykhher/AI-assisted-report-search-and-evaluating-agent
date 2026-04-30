"""Tune quality sub-weights using LLM-generated holistic quality labels.

Workflow:
  1. Load documents from the curated benchmark CSV.
  2. For each document, call local Qwen with a holistic quality prompt that
     asks for an overall credibility score with no sub-metric rubric (black box).
  3. Compute heuristic sub-scores (methodology, citation, consistency,
     structure, data_density, claim_density) using the existing extractor.
  4. Fit constrained least squares: find weights w (w >= 0, sum = 1) such
     that X @ w ≈ y_llm, where X is the sub-score matrix and y is the LLM labels.
  5. Report current vs. learned weights and optionally apply them to scoring.py.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_qwen import _generate, _local_qwen_enabled
from source.extractor import (
    bottom_reference_score,
    data_density,
    footnote_score,
    has_methodology,
    institution_score,
    compute_structure_score,
)
from source.scoring import compute_claim_density, compute_consistency_score
from source.runtime.curated_benchmark import _build_proxy_report_text

DATA_DIR = PROJECT_ROOT / "data" / "curated_benchmark"
SCORING_PATH = PROJECT_ROOT / "source" / "scoring.py"

QUALITY_FEATURES = [
    "methodology",
    "citation",
    "consistency",
    "structure",
    "data_density",
    "claim_density",
]
CURRENT_WEIGHTS = [0.22, 0.22, 0.18, 0.14, 0.14, 0.10]

_LLM_QUALITY_PROMPT = """\
You are an expert evaluator of analytical business and research reports.

Read the document excerpt below and rate its overall analytical quality and \
credibility on a scale from 0.0 to 1.0.

Guidelines:
- 1.0 = rigorous, well-evidenced, methodologically transparent research report
- 0.7 = solid report with clear findings but limited methodology detail
- 0.4 = mixed document with some useful content but weak evidence or structure
- 0.1 = promotional, unsupported, or lacking analytical substance
- 0.0 = not a report at all

Use your own holistic judgment. Do not follow any predefined rubric or formula.

Return ONLY valid JSON with no extra text:
{{"quality_score": <float between 0.0 and 1.0>}}

Document:
{text}"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_documents() -> list[dict[str, str]]:
    path = DATA_DIR / "documents.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        docs = list(csv.DictReader(fh))

    # Overlay real fetched text when available
    texts_path = DATA_DIR / "document_texts.csv"
    if texts_path.exists():
        with texts_path.open(newline="", encoding="utf-8") as fh:
            fetched = {r["doc_id"]: r for r in csv.DictReader(fh)}
        upgraded = 0
        for doc in docs:
            row = fetched.get(doc["doc_id"])
            if row and row.get("status") == "ok" and row.get("fetched_text", "").strip():
                doc["snippet"] = row["fetched_text"]
                upgraded += 1
        if upgraded:
            print(f"  [dataset] Replaced {upgraded}/100 snippets with real fetched text.")
    return docs


# ---------------------------------------------------------------------------
# LLM oracle — local Qwen or Claude API fallback
# ---------------------------------------------------------------------------

def _parse_score(raw: str) -> float | None:
    if not raw:
        return None
    match = re.search(r'"quality_score"\s*:\s*([0-9.]+)', raw)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return None


def _claude_quality_score(text: str) -> float | None:
    """Call Anthropic Claude API (requires ANTHROPIC_API_KEY env var)."""
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=64,
            messages=[{"role": "user", "content": _LLM_QUALITY_PROMPT.format(text=text[:3000])}],
        )
        raw = msg.content[0].text if msg.content else ""
        return _parse_score(raw)
    except Exception as exc:
        print(f"    [claude api error] {exc}")
        return None


_QWEN_SYSTEM_PROMPT = (
    "You are an expert evaluator of analytical business and research reports. "
    "When asked to rate a document, return ONLY valid JSON in the exact format: "
    '{{"quality_score": <number>}}  — no other text, no explanation.'
)

_QWEN_USER_TEMPLATE = """\
Rate the overall analytical quality and credibility of this document.

Output a single float between 0.0 and 1.0 based on your holistic judgment:
- Consider methodological rigor, evidence quality, citation density, and analytical depth.
- Use the full range: a rigorous research report should score near 1.0, a promotional \
landing page near 0.0, and everything else proportionally in between.
- Do NOT round to preset values. Produce a precise decimal like 0.63 or 0.41.

Return ONLY: {{"quality_score": <float>}}

Document:
{text}"""


def _llm_quality_score(text: str) -> float | None:
    # Try local Qwen first
    if _local_qwen_enabled():
        try:
            user_msg = _QWEN_USER_TEMPLATE.format(text=text[:2000])
            raw = _generate(user_msg, max_new_tokens=48, system_prompt=_QWEN_SYSTEM_PROMPT)
            score = _parse_score(raw)
            if score is not None:
                return score
        except Exception:
            pass
    # Fall back to Claude API
    return _claude_quality_score(text)


def collect_llm_labels(documents: list[dict[str, str]]) -> list[dict]:
    """Call LLM on each document and return rows with doc_id + llm_score."""
    results = []
    for i, doc in enumerate(documents):
        text = _build_proxy_report_text(doc)
        score = _llm_quality_score(text)
        status = f"{score:.3f}" if score is not None else "FAILED"
        print(f"  [{i+1:>3}/{len(documents)}] {doc.get('doc_id', '?'):<12} llm_score={status}", flush=True)
        results.append({
            "doc_id": doc.get("doc_id", ""),
            "document_type": doc.get("document_type", ""),
            "llm_quality_score": score,
            "text": text,
        })
    return results


# ---------------------------------------------------------------------------
# Heuristic sub-scores
# ---------------------------------------------------------------------------

def _citation_score(text: str) -> float:
    score = max(bottom_reference_score(text), footnote_score(text))
    score += 0.2 * institution_score(text)
    return min(score, 1.0)


def compute_heuristic_subscores(text: str) -> dict[str, float]:
    return {
        "methodology": float(has_methodology(text)),
        "citation": _citation_score(text),
        "consistency": compute_consistency_score(text),
        "structure": compute_structure_score(text),
        "data_density": data_density(text),
        "claim_density": compute_claim_density(text),
    }


# ---------------------------------------------------------------------------
# Constrained regression
# ---------------------------------------------------------------------------

def _fit_weights(X: np.ndarray, y: np.ndarray, w0: list[float]) -> tuple[np.ndarray, float, float]:
    def mse(w: np.ndarray) -> float:
        return float(np.mean((X @ w - y) ** 2))

    result = minimize(
        mse,
        np.array(w0, dtype=float),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(w0),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 5000},
    )
    return result.x, mse(np.array(w0)), mse(result.x)


def _r2(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    ss_res = np.sum((y - X @ w) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Apply to scoring.py
# ---------------------------------------------------------------------------

def _apply_weights(tuned: np.ndarray) -> None:
    names = ["methodology", "citation", "consistency", "structure", "data_support", "claim_density"]
    source = SCORING_PATH.read_text(encoding="utf-8")
    lines = "    score = (\n"
    for i, (name, w) in enumerate(zip(names, tuned)):
        prefix = "        " if i == 0 else "        + "
        lines += f"{prefix}{round(float(w), 4)} * {name}\n"
    lines += "    )"
    source = re.sub(
        r"    score = \(\s*0\.\d+ \* methodology.*?\)",
        lines,
        source,
        flags=re.DOTALL,
    )
    SCORING_PATH.write_text(source, encoding="utf-8")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_results(
    features: list[str],
    current_w: list[float],
    tuned_w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    mse_cur: float,
    mse_tun: float,
) -> None:
    col = 14
    print(f"\n{'=' * 52}")
    print("  Quality weights tuned from LLM labels")
    print(f"{'=' * 52}")
    print(f"  {'Feature':<{col}} {'Current':>9} {'Tuned':>9} {'Delta':>9}")
    print(f"  {'-' * (col + 29)}")
    for feat, cw, tw in zip(features, current_w, tuned_w):
        delta = float(tw) - cw
        sign = "+" if delta > 0.001 else ("-" if delta < -0.001 else " ")
        print(f"  {feat:<{col}} {cw:>9.4f} {float(tw):>9.4f} {sign}{abs(delta):>8.4f}")
    print(f"  {'-' * (col + 29)}")
    print(f"  {'Sum':<{col}} {sum(current_w):>9.4f} {tuned_w.sum():>9.4f}")
    print(f"\n  MSE  current  : {mse_cur:.6f}")
    print(f"  MSE  tuned    : {mse_tun:.6f}")
    print(f"  R2   current  : {_r2(X, y, np.array(current_w)):.4f}")
    print(f"  R2   tuned    : {_r2(X, y, tuned_w):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import os
    has_qwen = _local_qwen_enabled()
    has_claude = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
    if not has_qwen and not has_claude:
        print("ERROR: No LLM backend available.")
        print("  Option 1: set USE_LOCAL_QWEN=true (requires local Qwen model)")
        print("  Option 2: set ANTHROPIC_API_KEY=<your key>")
        sys.exit(1)
    backend = "local Qwen" if has_qwen else "Claude API (claude-haiku-4-5)"
    print(f"LLM backend: {backend}")

    documents = _load_documents()
    print(f"Loaded {len(documents)} documents. Collecting LLM quality labels ...\n")

    labeled = collect_llm_labels(documents)
    valid = [row for row in labeled if row["llm_quality_score"] is not None]
    failed = len(labeled) - len(valid)
    print(f"\n{len(valid)} scored, {failed} failed.")

    if len(valid) < 10:
        print("Too few valid labels to fit weights reliably. Exiting.")
        sys.exit(1)

    # Save labels for inspection
    label_path = DATA_DIR / "llm_quality_labels.csv"
    with label_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["doc_id", "document_type", "llm_quality_score"])
        writer.writeheader()
        for row in labeled:
            writer.writerow({k: row[k] for k in ["doc_id", "document_type", "llm_quality_score"]})
    print(f"Labels saved to {label_path.name}")

    # Build feature matrix
    X_rows = []
    y_vals = []
    for row in valid:
        subs = compute_heuristic_subscores(row["text"])
        X_rows.append([subs[f] for f in QUALITY_FEATURES])
        y_vals.append(row["llm_quality_score"])

    X = np.array(X_rows, dtype=float)
    y = np.array(y_vals, dtype=float)

    print(f"\nFitting constrained weights on {len(y)} samples ...")
    tuned_w, mse_cur, mse_tun = _fit_weights(X, y, CURRENT_WEIGHTS)

    _print_results(QUALITY_FEATURES, CURRENT_WEIGHTS, tuned_w, X, y, mse_cur, mse_tun)

    # Ask before applying
    print("\nApply tuned weights to source/scoring.py? [y/N] ", end="", flush=True)
    answer = input().strip().lower()
    if answer == "y":
        _apply_weights(tuned_w)
        print("scoring.py updated.")
    else:
        print("Weights not applied.")


if __name__ == "__main__":
    main()
