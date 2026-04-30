# Industry Report Discovery Agent

An agent for finding, filtering, scoring, and verifying industry and research reports. Given a user query, it searches the web via SerpApi, classifies candidates, scores quality across interpretable dimensions, verifies claims, and returns a ranked list with explanations.

## What It Does

Given a query like `"enterprise AI adoption report 2025"`, the agent:

1. interprets the request into a structured plan
2. searches the web via SerpApi Google Search
3. normalises and filters candidate documents
4. fetches and parses fuller document text when possible
5. classifies source authority and document type
6. extracts credibility and analytical quality signals
7. scores and ranks candidate reports
8. reflects on result quality and replans if results are weak
9. extracts claims and citations from top reports
10. runs deterministic lightweight claim verification
11. recomputes verification-adjusted quality scores
12. reranks verified results

The verification layer is intentionally lightweight and LLM-free. It checks local report evidence, extracted citation URLs, numeric matches, source reliability, and claim/context overlap — making the behavior explainable and auditable.

## System Flow

```text
User query
  → query planner
  → agent state
  → controller loop
      → search / fetch / parse / classify / score / rank
      → reflect
      → replan and repeat if quality is weak
  → verify top reports
  → recompute verification-adjusted quality
  → rerank
  → final ranked results
```

Reflection diagnoses failures such as topic drift, weak quality signals, too few results, low authority, stale results, or fetch failures. The agent replans and searches again until quality is sufficient or the replan budget is exhausted.

## Project Layout

```text
agent/
  main.py                         entry point (wraps source/main.py)
  local_qwen.py                   optional local Qwen integration
  .env                            API keys (not committed)
  requirements.txt
  README.md
  source/
    main.py                       CLI runner
    agent.py                      orchestration entrypoints
    controller.py                 LLM-guided controller loop and replan logic
    agent_state.py                shared agent state and candidate records
    tool_registry.py              registered tool definitions
    reflection.py                 quality reflection and failure diagnosis
    search.py                     SerpApi search with local fallback
    API.py                        SerpApi HTTP wrapper
    filter.py                     weak result filtering
    extractor.py                  text signal extraction
    scoring.py                    composite quality scoring
    query/
      handler.py                  query normalisation and expansion
      planner.py                  structured query planning
    classifier/
      report_classifier.py        document type classification
      source_classifier.py        source authority classification
    fetching/
      document_fetcher.py         HTTP fetch and HTML/PDF text extraction
      text_parser.py              parse full text into sections and quality flags
      parser.py                   parse SerpApi JSON responses
    verification/
      core.py                     claim extraction and verification orchestration
      claims.py                   deterministic claim verifier
      citations.py                URL and reference extraction
      metrics.py                  verification metric computation
    runtime/
      curated_benchmark.py        benchmark runner and weight tuner
      evaluate_agent.py           agent evaluation harness
      iteration_controller.py     query rewriting on failure
      exporter.py                 JSON/CSV export
      schemas.py                  result dataclasses
  data/
    curated_benchmark/
      queries.csv                 20 representative queries
      documents.csv               5 candidate documents per query (100 total)
      document_texts.csv          fetched full text per document
      retrieval_annotations.csv   relevance and result-class labels
      quality_annotations.csv     DEER-inspired quality labels
      llm_quality_labels.csv      Qwen holistic quality scores (oracle)
      tuning_results.json         recorded weight-tuning run results
      README.md
    dataset.json
  scripts/
    update_document_urls.py       refresh document URLs via SerpApi
    annotate_benchmark.py         annotate benchmark with Qwen + heuristic labels
    annotate_llm_score_labels.py  generate Qwen holistic quality labels
    tune_final_from_llm_scores.py fit final-score coefficients to Qwen oracle
  test/
    test_dataset_scores.py        score all benchmark documents and report metrics
    test_query_dataset.py         test query planning and search on benchmark queries
    tune_from_llm_labels.py       Qwen oracle weight tuning (constrained regression)
    tune_weights.py               weight tuning against quality_annotations.csv
    fetch_document_texts.py       fetch and cache full text for benchmark documents
    rebuild_benchmark_documents.py rebuild documents.csv from live SerpApi results
```

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

For local Qwen inference (optional, requires a downloaded `Qwen2.5-*-Instruct` model):

```bash
pip install transformers torch accelerate bitsandbytes protobuf sentencepiece
```

For benchmark tuning scripts:

```bash
pip install numpy scipy anthropic ddgs
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `SERPAPI_API_KEY` | Required for live web search |
| `USE_LOCAL_QWEN` | Set to `1` to enable local Qwen inference |
| `QWEN_MODEL_PATH` | Override the Qwen model directory |
| `QWEN_EXTRA_SITE_PACKAGES` | Point to a venv site-packages with torch/transformers |

The recommended way to set `SERPAPI_API_KEY` is a `.env` file at the project root — it is loaded automatically at startup and is already listed in `.gitignore`.

```bash
# .env
SERPAPI_API_KEY=your_key_here
```

## Running the Agent

```bash
# Basic
python source/main.py "enterprise AI adoption benchmark report 2025"

# With debug state
python source/main.py "enterprise AI adoption benchmark report 2025" --debug

# JSON output
python source/main.py "enterprise AI adoption benchmark report 2025" --json

# Control verification depth
python source/main.py "enterprise AI adoption benchmark report 2025" --verify-top-n 3

# Suppress step-by-step logs
python source/main.py "enterprise AI adoption benchmark report 2025" --quiet
```

On Windows with a local venv:

```powershell
.\.venv\Scripts\python.exe source\main.py "enterprise AI adoption benchmark report 2025"
```

## Example Output

```text
Query: enterprise AI adoption benchmark report 2025
Stop reason: sufficient_quality
Processing time: 1842 ms

Top Ranked Reports
--------------------------------------------------------------------------------
1. State of Enterprise AI Adoption Report 2025
   score=0.781  type=report  source=https://example.com/report.pdf
   summary: This report covers state of enterprise ai adoption. It is published by McKinsey.
   why: contains methodology, strong citation support, internally consistent analysis
   verify: high confidence - Adoption increased by 35% year-over-year in enterprise deployments.
```

## Scoring

### Final Score

```text
final_score =
  0.35 × relevance_score
+ 0.20 × report_validity_score
+ 0.30 × quality_score
+ 0.15 × authority_score
```

### Quality Sub-scores

Six components blended into `quality_score` (tuned weights, applied 2026-04-30):

| Component | Weight | What it measures |
|---|---|---|
| structure | 0.355 | Report-like section organisation |
| claim_density | 0.273 | Verifiable analytical claims |
| consistency | 0.131 | Coherent claims backed by data |
| methodology | 0.100 | Research design, survey process, benchmark methodology |
| data_density | 0.079 | Non-year numeric density |
| citation | 0.062 | Citations, footnotes, named institutional sources |

Weights were fit by constrained regression against a Qwen 7B holistic quality oracle on 100 benchmark documents (see `data/curated_benchmark/tuning_results.json`). Methodology is floored at 0.100 — its tuned-zero value reflected web-page text rather than real report PDFs.

### Relevance

Measures query-topic overlap, ignoring generic terms like `pdf`, `report`, `industry`, `forecast`.

### Report Validity

Estimates whether a candidate is a structured report rather than a blog, landing page, or news snippet. Signals: document length, PDF format, section headings, methodology and reference language.

### Authority

Classifies the publishing source: intergovernmental body, consulting firm, research provider, trade media, government, academic, or vendor.

## Verification and Reranking

The verification pipeline lives in `source/verification/`:

- `core.py` — claim extraction, orchestration, and `attach_verification_notes()`
- `citations.py` — URL and numbered-reference extraction
- `claims.py` — deterministic lexical/numeric claim verification
- `metrics.py` — 12 verification metrics (accuracy, coverage, diversity, reliability)

Verification metrics are blended back into quality sub-components with bounded weights:

```text
methodology = 80% original + 20% verification
citation    = 65% original + 35% verification
consistency = 70% original + 30% verification
structure   = 85% original + 15% verification
data        = 70% original + 30% verification
claim       = 70% original + 30% verification
```

## Curated Benchmark

The benchmark lives in [`data/curated_benchmark/`](data/curated_benchmark/). It contains 20 queries, 5 candidate documents per query (100 total), relevance/ranking labels, DEER-inspired quality labels, and Qwen holistic oracle scores.

### Refresh document URLs

Searches SerpApi for each query and replaces stale URLs in `documents.csv`:

```bash
python scripts/update_document_urls.py
# Preview without writing
python scripts/update_document_urls.py --dry-run
```

### Re-annotate with Qwen

```bash
# Requires USE_LOCAL_QWEN=1 and a downloaded Qwen2.5-*-Instruct model.
USE_LOCAL_QWEN=1 python scripts/annotate_benchmark.py

# Single query
USE_LOCAL_QWEN=1 python scripts/annotate_benchmark.py --query q003

# Preview without saving
USE_LOCAL_QWEN=1 python scripts/annotate_benchmark.py --dry-run
```

### Weight tuning

```bash
# Tune quality sub-score weights against Qwen holistic oracle labels
python test/tune_from_llm_labels.py

# Tune final-score component weights against Qwen oracle labels
python scripts/tune_final_from_llm_scores.py
```

### Benchmark evaluation

```powershell
# Score all benchmark documents
.\.venv\Scripts\python.exe test\test_dataset_scores.py

# Run full benchmark evaluation
.\.venv\Scripts\python.exe source\runtime\curated_benchmark.py --rank

# Tune final score weights via benchmark
.\.venv\Scripts\python.exe source\runtime\curated_benchmark.py --tune --top-n 10
```

## Local Qwen Integration

`local_qwen.py` provides optional on-device inference using a Hugging Face `Qwen2.5-*-Instruct` model (7B recommended; loaded in 4-bit on CUDA when available). It is used for:

- **`suggest_search_queries`** / **`rewrite_search_query`** — improved search queries for a given topic
- **`assess_text_signals`** — additional credibility signals fed into the scoring pipeline (opt-in via `USE_LOCAL_QWEN_SIGNALS=1`)

Enable with:

```bash
USE_LOCAL_QWEN=1 python source/main.py "..."

# Point to a specific model
QWEN_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct-1M USE_LOCAL_QWEN=1 python source/main.py "..."
```

The module discovers models and compatible venvs automatically. If inference fails, all functions return safe defaults and the pipeline continues without disruption.

## Design Notes

- **Search is swappable** — SerpApi and a local curated fallback are the only places where document sources enter the pipeline.
- **Scoring is weighted and interpretable** — each sub-component has a documented formula and can be tuned independently against oracle labels.
- **Verification is deterministic** — no LLM, no external calls beyond what was already fetched.
- **Reflection and replanning** are controller-level behaviors separate from scoring logic.
- **The benchmark is small but real** — 20 curated queries, live report URLs, Qwen-annotated quality labels.
