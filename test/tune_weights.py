"""Tune quality and final-score weights using ranking-based objectives.

Stage 1 — quality sub-weights:
    Grid search over [methodology, citation, consistency, structure,
    data_density, claim_density] weights. Objective: pairwise accuracy
    when using quality sub-scores alone to rank the 5 document types
    in each query group (research_report > whitepaper > archived_pdf >
    news_article > landing_page).

Stage 2 — final score weights:
    Uses curated_benchmark.tune_final_weights() which already implements
    a grid search over [relevance, report_validity, quality, authority]
    using NDCG@5 + pairwise accuracy + top-1 accuracy as the objective.

Prints current vs. best weights and updates scoring.py automatically.
"""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.runtime.curated_benchmark import tune_final_weights

DATA_DIR = PROJECT_ROOT / "data" / "curated_benchmark"
SCORING_PATH = PROJECT_ROOT / "source" / "scoring.py"

QUALITY_FEATURES = [
    "methodology_score",
    "citation_score",
    "consistency_score",
    "structure_score",
    "data_density",
    "claim_density",
]
CURRENT_QUALITY_WEIGHTS = [0.22, 0.22, 0.18, 0.14, 0.14, 0.10]
CURRENT_FINAL_WEIGHTS = {
    "relevance_score": 0.35,
    "report_validity_score": 0.20,
    "quality_score": 0.30,
    "authority_score": 0.15,
}


# ---------------------------------------------------------------------------
# Load CSVs
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Stage 1 — quality weight grid search
# ---------------------------------------------------------------------------

def _quality_score(row: dict[str, str], weights: list[float]) -> float:
    return sum(w * _float(row[f]) for w, f in zip(weights, QUALITY_FEATURES))


def _pairwise_acc_quality(
    quality_rows: list[dict[str, str]],
    retrieval_rows: list[dict[str, str]],
    weights: list[float],
) -> float:
    pref_by_doc = {r["doc_id"]: _int(r["ranking_preference"]) for r in retrieval_rows}

    # Group docs by query prefix (d001_*, d002_*, ...)
    groups: dict[str, list[dict[str, str]]] = {}
    for row in quality_rows:
        prefix = row["doc_id"][:4]  # e.g. "d001"
        groups.setdefault(prefix, []).append(row)

    correct = 0
    total = 0
    for group in groups.values():
        for a, b in itertools.combinations(group, 2):
            pa = pref_by_doc.get(a["doc_id"], 99)
            pb = pref_by_doc.get(b["doc_id"], 99)
            if pa == pb:
                continue
            qa = _quality_score(a, weights)
            qb = _quality_score(b, weights)
            expected_higher = a if pa < pb else b
            actual_higher = a if qa >= qb else b
            correct += int(expected_higher["doc_id"] == actual_higher["doc_id"])
            total += 1

    return correct / total if total else 0.0


def _iter_quality_grid(step: float = 0.05) -> list[list[float]]:
    """All 6-weight vectors >= 0 that sum to 1 at the given step size."""
    n = len(QUALITY_FEATURES)
    scale = round(1.0 / step)
    grids: list[list[float]] = []

    def recurse(remaining: int, budget: int, current: list[float]) -> None:
        if remaining == 1:
            w = budget * step
            if w > 0:
                grids.append(current + [round(w, 4)])
            return
        for val in range(1, budget - (remaining - 1)):
            recurse(remaining - 1, budget - val, current + [round(val * step, 4)])

    recurse(n, scale, [])
    return grids


def tune_quality_weights(step: float = 0.05) -> tuple[list[float], float, float]:
    quality_rows = _load_csv(DATA_DIR / "quality_annotations.csv")
    retrieval_rows = _load_csv(DATA_DIR / "retrieval_annotations.csv")

    current_acc = _pairwise_acc_quality(quality_rows, retrieval_rows, CURRENT_QUALITY_WEIGHTS)

    best_weights = CURRENT_QUALITY_WEIGHTS[:]
    best_acc = current_acc
    for weights in _iter_quality_grid(step):
        acc = _pairwise_acc_quality(quality_rows, retrieval_rows, weights)
        if acc > best_acc:
            best_acc = acc
            best_weights = weights

    return best_weights, current_acc, best_acc


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_quality(
    features: list[str],
    current_w: list[float],
    tuned_w: list[float],
    current_acc: float,
    tuned_acc: float,
) -> None:
    col = 24
    print(f"\n{'=' * 56}")
    print("  Stage 1 — Quality sub-weights (pairwise accuracy)")
    print(f"{'=' * 56}")
    print(f"  {'Feature':<{col}} {'Current':>9} {'Tuned':>9} {'Delta':>9}")
    print(f"  {'-' * (col + 29)}")
    for feat, cw, tw in zip(features, current_w, tuned_w):
        delta = tw - cw
        arrow = "+" if delta > 0.001 else ("-" if delta < -0.001 else " ")
        print(f"  {feat:<{col}} {cw:>9.4f} {tw:>9.4f} {arrow}{abs(delta):>8.4f}")
    print(f"  {'-' * (col + 29)}")
    print(f"  {'Sum':<{col}} {sum(current_w):>9.4f} {sum(tuned_w):>9.4f}")
    print(f"\n  Pairwise accuracy (current): {current_acc:.4f}")
    print(f"  Pairwise accuracy (tuned)  : {tuned_acc:.4f}")


def _print_final(
    current_w: dict[str, float],
    best: dict[str, Any],
) -> None:
    col = 24
    tw = best["weights"]
    print(f"\n{'=' * 56}")
    print("  Stage 2 — Final score weights (NDCG+pairwise+top1)")
    print(f"{'=' * 56}")
    print(f"  {'Feature':<{col}} {'Current':>9} {'Tuned':>9} {'Delta':>9}")
    print(f"  {'-' * (col + 29)}")
    for feat in current_w:
        cw = current_w[feat]
        tw_val = tw[feat]
        delta = tw_val - cw
        arrow = "+" if delta > 0.001 else ("-" if delta < -0.001 else " ")
        print(f"  {feat:<{col}} {cw:>9.4f} {tw_val:>9.4f} {arrow}{abs(delta):>8.4f}")
    print(f"  {'-' * (col + 29)}")
    print(f"  {'Sum':<{col}} {sum(current_w.values()):>9.4f} {sum(tw.values()):>9.4f}")
    print(f"\n  Objective (current): (see below)")
    print(f"  Objective (tuned)  : {best['objective']:.4f}")
    print(f"  NDCG@5             : {best['mean_ndcg_at_k']:.4f}")
    print(f"  Pairwise accuracy  : {best['mean_pairwise_accuracy']:.4f}")
    print(f"  Top-1 accuracy     : {best['top1_accuracy']:.4f}")


# ---------------------------------------------------------------------------
# Apply to scoring.py
# ---------------------------------------------------------------------------

def _apply_to_scoring(quality_weights: list[float], final_weights: dict[str, float]) -> None:
    source = SCORING_PATH.read_text(encoding="utf-8")

    # Build new FINAL_WEIGHTS block
    fw_lines = "FINAL_WEIGHTS: dict[str, float] = {\n"
    for key, val in final_weights.items():
        fw_lines += f'    "{key}": {round(val, 4)},\n'
    fw_lines += "}"

    # Build new quality formula inside compute_quality_score
    names = ["methodology", "citation", "consistency", "structure", "data_support", "claim_density"]
    q_lines = "    score = (\n"
    for name, w in zip(names, quality_weights):
        q_lines += f"        {round(w, 4)} * {name}\n"
        if name != names[-1]:
            q_lines = q_lines.rstrip("\n") + "\n"
    # rebuild as a proper sum
    q_parts = [f"        {round(w, 4)} * {name}" for name, w in zip(names, quality_weights)]
    q_lines = "    score = (\n" + "\n        + ".join(q_parts).replace("        + ", "        + ", 1) + "\n    )"
    # simpler rebuild:
    q_lines = "    score = (\n"
    for i, (name, w) in enumerate(zip(names, quality_weights)):
        prefix = "        " if i == 0 else "        + "
        q_lines += f"{prefix}{round(w, 4)} * {name}\n"
    q_lines += "    )"

    # Replace FINAL_WEIGHTS dict in source
    import re
    source = re.sub(
        r'FINAL_WEIGHTS: dict\[str, float\] = \{[^}]+\}',
        fw_lines,
        source,
        flags=re.DOTALL,
    )

    # Replace quality score formula (the score = ( ... ) block)
    source = re.sub(
        r'    score = \(\s*0\.\d+ \* methodology.*?\)',
        q_lines,
        source,
        flags=re.DOTALL,
    )

    SCORING_PATH.write_text(source, encoding="utf-8")
    print(f"\n  scoring.py updated.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Tuning quality sub-weights (step=0.05) ...")
    tuned_q, acc_q_cur, acc_q_tun = tune_quality_weights(step=0.05)
    _print_quality(QUALITY_FEATURES, CURRENT_QUALITY_WEIGHTS, tuned_q, acc_q_cur, acc_q_tun)

    print("\nTuning final score weights (step=0.05) ...")
    top_results = tune_final_weights(step=0.05, k=5, top_n=1)
    best_final = top_results[0]
    _print_final(CURRENT_FINAL_WEIGHTS, best_final)

    print(f"\n{'=' * 56}")
    print("  Applying tuned weights to source/scoring.py ...")
    _apply_to_scoring(tuned_q, best_final["weights"])

    print("\n  Done. Suggested weights:")
    q_str = ", ".join(f"{f}: {round(w, 4)}" for f, w in zip(QUALITY_FEATURES, tuned_q))
    f_str = ", ".join(f"{k}: {round(v, 4)}" for k, v in best_final["weights"].items())
    print(f"  Quality  -> {q_str}")
    print(f"  Final    -> {f_str}")


if __name__ == "__main__":
    main()
