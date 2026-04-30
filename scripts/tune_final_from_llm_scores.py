"""Tune final-score coefficients against local-Qwen oracle labels.

Reads:
    data/curated_benchmark/quality_annotations.csv
    data/curated_benchmark/llm_score_labels.csv

The fitted model is:
    llm_final_score ~= w1*relevance_score
                    + w2*report_validity_score
                    + w3*quality_score
                    + w4*authority_score

Weights are constrained to be non-negative and sum to 1. This script reports
the best grid-search weights but does not edit source/scoring.py.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "curated_benchmark"

FEATURES = [
    "relevance_score",
    "report_validity_score",
    "quality_score",
    "authority_score",
]

CURRENT_WEIGHTS = {
    "relevance_score": 0.35,
    "report_validity_score": 0.20,
    "quality_score": 0.30,
    "authority_score": 0.15,
}


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _predict(row: dict[str, str], weights: dict[str, float]) -> float:
    return sum(weights[feature] * _float(row.get(feature)) for feature in FEATURES)


def _mse(rows: list[dict[str, str]], labels: dict[str, dict[str, str]], weights: dict[str, float]) -> float:
    errors: list[float] = []
    for row in rows:
        label = labels.get(row["doc_id"])
        if not label:
            continue
        target = _float(label.get("llm_final_score"))
        pred = _predict(row, weights)
        errors.append((pred - target) ** 2)
    return sum(errors) / len(errors) if errors else 0.0


def _iter_weights(step: float) -> list[dict[str, float]]:
    scale = round(1.0 / step)
    vectors: list[dict[str, float]] = []
    for parts in itertools.product(range(scale + 1), repeat=len(FEATURES)):
        if sum(parts) != scale:
            continue
        weights = {feature: round(part * step, 4) for feature, part in zip(FEATURES, parts)}
        vectors.append(weights)
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune final weights from LLM score labels.")
    parser.add_argument("--step", type=float, default=0.05, help="Grid step size, default 0.05.")
    args = parser.parse_args()

    quality_rows = _load_csv(DATA_DIR / "quality_annotations.csv")
    label_rows = _load_csv(DATA_DIR / "llm_score_labels.csv")
    labels = {row["doc_id"]: row for row in label_rows}

    current_mse = _mse(quality_rows, labels, CURRENT_WEIGHTS)
    best_weights = CURRENT_WEIGHTS
    best_mse = current_mse

    for weights in _iter_weights(args.step):
        mse = _mse(quality_rows, labels, weights)
        if mse < best_mse:
            best_mse = mse
            best_weights = weights

    print("Final-score coefficient tuning against llm_final_score")
    print(f"Rows: {len(quality_rows)} quality rows, {len(label_rows)} LLM labels")
    print(f"Current MSE: {current_mse:.6f}")
    print(f"Best MSE   : {best_mse:.6f}")
    print("\nWeights:")
    for feature in FEATURES:
        current = CURRENT_WEIGHTS[feature]
        tuned = best_weights[feature]
        print(f"  {feature:<24} current={current:.4f} tuned={tuned:.4f}")

    print("\nLLM label calibration:")
    for score_field, label_field in [
        ("report_validity_score", "llm_validity_score"),
        ("quality_score", "llm_quality_score"),
    ]:
        errors = []
        for row in quality_rows:
            label = labels.get(row["doc_id"])
            if label:
                errors.append((_float(row[score_field]) - _float(label[label_field])) ** 2)
        mse = sum(errors) / len(errors) if errors else 0.0
        print(f"  {score_field:<24} vs {label_field:<18} MSE={mse:.6f}")


if __name__ == "__main__":
    main()
