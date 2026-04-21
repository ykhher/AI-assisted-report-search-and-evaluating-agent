# Curated Report Verification Benchmark

This folder contains a small synthetic benchmark for an AI-assisted report search and evaluation agent.
It is designed for a capstone project: small enough to inspect manually, structured enough to evaluate retrieval, ranking, report quality, and lightweight verification behavior.

## Files

- `queries.csv`: representative user queries for report discovery.
- `documents.csv`: five retrieved candidate documents per query. This file contains retrieval metadata only, not quality labels.
- `retrieval_annotations.csv`: query-document relevance labels and result classes.
- `quality_annotations.csv`: document-level report quality labels using compressed DEER-inspired dimensions.

## Label Scale

All quality and relevance labels use the same interpretable 0-3 scale:

- `0`: absent, poor, or not useful
- `1`: weak
- `2`: acceptable
- `3`: strong

`ranking_preference` uses `1` for the best candidate within a query and `5` for the weakest.

## Compressed DEER-Inspired Dimensions

DEER is used only as an expert-informed annotation framework. The benchmark compresses it into four practical dimensions:

- `deer_method_label`: whether the document explains method, sample, scope, or analytical approach.
- `deer_evidence_label`: whether claims are supported by data, references, charts, or source notes.
- `deer_transparency_label`: whether assumptions, definitions, limitations, or sources are visible.
- `deer_recency_label`: whether the document is current enough for the query.

The dataset intentionally does not copy the full DEER rubric.

## Intended Uses

- Retrieval evaluation: did the agent find report-like candidates?
- Ranking evaluation: did strong reports rank above weak pages and false positives?
- Quality scoring calibration: do scoring coefficients align with human-readable labels?
- Verification support evaluation: does the selected report expose enough evidence for lightweight claim checks?

## Synthetic Data Note

The rows are synthetic starter examples. URLs use `https://benchmark.local/...` so they cannot be confused with live documents.
Replace or extend rows with real search results when you are ready for a stronger benchmark.
