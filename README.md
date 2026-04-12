# Industry Report Ranking Agent

## Overview

This project is a Python agent that finds report-like documents, scores credibility, and returns ranked results with explanations.

The system is designed for practical retrieval and ranking. It is not a full fact-checking engine.

## What The System Does

For each query, the agent:
1. rewrites/expands the query (optionally with local Qwen)
2. retrieves candidates (API-first, with local fallback)
3. filters non-report or weak candidates
4. extracts report-quality signals from snippet/full text + metadata
5. computes component scores and final rank score
6. returns ranked results and optional exports

## Scope And Non-Goals

In scope:
- report-style retrieval
- credibility-oriented scoring
- explainable ranking output

Out of scope:
- claim-level fact verification against external sources
- sentence-level truth labeling
- legal/compliance-grade auditing

## Current Architecture

Main flow:

```text
user query
-> query rewrite / refinement
-> search (API-first)
-> filter
-> optional fetch + parse enrichment
-> signal extraction
-> scoring + ranking
-> structured results / export
```

Agent loop behavior:
- runs one API retrieval call
- re-scores cached results across iterations
- diagnoses failure types and rewrites query when needed
- stops early when top quality is good enough

## Repository Layout (Current)

```text
agent/
  main.py
  local_qwen.py
  README.md
  requirements.txt
  SYSTEM_PROMPT.md
  source/
    agent.py
    main.py
    pipeline.py
    search.py
    API.py
    filter.py
    extractor.py
    scoring.py
    query_handler.py
    ranking.py
    classifier/
      source_classifier.py
      report_classifier.py
    fetching/
      document_fetcher.py
      text_parser.py
      parser.py
    runtime/
      evaluate_agent.py
      iteration_controller.py
      exporter.py
      schemas.py
  data/
    dataset.json
    benchmark_queries.csv
    benchmark_labels.json
  test/
    test_dataset_scores.py
```

## Key Modules

- `main.py`: root CLI entrypoint, calls `source.agent.agent_pipeline`
- `source/agent.py`: iterative orchestration, structured output, optional export
- `source/search.py`: API search + fallback search + URL deduplication
- `source/extractor.py`: text/metadata signal extraction
- `source/scoring.py`: component scores and final ranking logic
- `source/classifier/*`: source authority and report-type classifiers
- `source/fetching/*`: optional full-text fetch/parse utilities
- `source/runtime/evaluate_agent.py`: benchmark runner
- `local_qwen.py`: optional local LLM integration

## Scoring Model

The ranker computes interpretable score components and then combines them:

- `relevance_score`
- `report_validity_score`
- `quality_score`
- `authority_score`
- `final_score`

### Quality Score Formula

`quality_score` is computed in `source/scoring.py` as:

```text
quality_score =
  0.22 * methodology +
  0.22 * citation +
  0.18 * consistency +
  0.14 * structure +
  0.14 * data_support +
  0.10 * claim_density
```

Weights in current scoring logic (`source/scoring.py`):

```text
final_score =
  0.40 * relevance_score +
  0.20 * report_validity_score +
  0.25 * quality_score +
  0.15 * authority_score
```

## Local Qwen (Optional)

Local Qwen is optional. If unavailable, the system still runs with heuristic scoring.

Useful env vars:
- `USE_LOCAL_QWEN_SIGNALS=0` to disable local LLM scoring
- `QWEN_MODEL_PATH` to point to a local model directory
- `QWEN_EXTRA_SITE_PACKAGES` to add external site-packages path

## Installation

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
- core HTTP dependency (`requests`)
- tokenizer/runtime helpers (`protobuf`, `sentencepiece`)
- optional local-LLM runtime (`transformers`, `torch`)

## Run

Main agent:

```bash
python main.py
```

You will be prompted for a query and receive ranked JSON output.

## Benchmark Evaluation

Run benchmark with explicit data paths:

```bash
python -m source.runtime.evaluate_agent \
  --queries data/benchmark_queries.csv \
  --labels data/benchmark_labels.json \
  --output outputs/benchmark_results.json \
  --k 5
```

## Notes

- `SYSTEM_PROMPT.md` defines the strict JSON scoring prompt used by local Qwen scoring paths.
- If API retrieval fails, search falls back to local mock index logic in `source/search.py`.
- `doc/README_SCOPE.md` now points here; this file is the canonical project description.
