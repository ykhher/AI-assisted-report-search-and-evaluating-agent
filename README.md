# Industry Report Discovery Agent

An agent for finding, filtering, scoring, and verifying industry and research reports. Given a user query, it searches the web, classifies candidates, scores quality across interpretable dimensions, verifies claims, and returns a ranked list with explanations.

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
  → planner
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
  README.md
  requirements.txt
  local_qwen.py                   optional local Qwen integration
  source/
    main.py                       CLI runner
    agent.py                      agent loop
    controller.py                 controller and replan logic
    planner.py                    structured query planning
    agent_state.py                shared agent state
    tool_registry.py              registered tool definitions
    reflection.py                 quality reflection and diagnosis
    verification.py               claim and citation verification
    verification_metrics.py       verification metric computation
    claim_verifier.py             deterministic claim checker
    citation_extractor.py         URL and reference extraction
    search.py                     search helpers and fallback
    API.py                        SerpApi wrapper
    filter.py                     weak result filtering
    extractor.py                  text signal extraction
    scoring.py                    composite quality scoring
    query_handler.py              query rewriting and expansion
    pipeline.py                   end-to-end pipeline wiring
    ranking.py                    final ranker
    classifier/
      source_classifier.py
      report_classifier.py
    fetching/
      document_fetcher.py
      text_parser.py
      parser.py
    runtime/
      curated_benchmark.py        benchmark runner and tuner
      evaluate_agent.py           agent evaluation harness
      iteration_controller.py
      exporter.py
      schemas.py
  data/
    dataset.json
    benchmark_queries.csv
    benchmark_labels.json
    curated_benchmark/
      queries.csv                 20 representative queries
      documents.csv               5 candidate documents per query (real URLs)
      retrieval_annotations.csv   relevance and result-class labels
      quality_annotations.csv     DEER-inspired quality labels
      README.md
  scripts/
    update_document_urls.py       renew document URLs via SerpApi search
    annotate_benchmark.py         annotate benchmark with Qwen + heuristic
```

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Optional local Qwen support requires a downloaded `Qwen2.5-*-Instruct` model in the Hugging Face cache (`~/.cache/huggingface/hub/`). The main pipeline is fully deterministic without it.

## Environment Variables

| Variable | Purpose |
|---|---|
| `SERPAPI_API_KEY` | Required for live web search |
| `USE_LOCAL_QWEN` | Set to `1` to enable local Qwen inference |
| `QWEN_MODEL_PATH` | Override the Qwen model directory |
| `QWEN_EXTRA_SITE_PACKAGES` | Point to a venv site-packages with torch/transformers |

Do not commit API keys to the repository.

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
────────────────────────────────────────────────────────────────────────────────
1. State of Enterprise AI Adoption Report 2025
   score=0.781  type=report  source=https://example.com/report.pdf
   summary: Covers enterprise AI adoption trends for 2025.
   why: contains methodology, strong citation support, internally consistent analysis
   verify: high confidence — Adoption increased by 35% in 2025.
```

## Scoring

The ranker uses four interpretable components:

```text
final_score =
  0.35 × relevance_score
+ 0.20 × report_validity_score
+ 0.30 × quality_score
+ 0.15 × authority_score
```

### Relevance

Measures query-topic overlap, ignoring generic terms like `pdf`, `report`, `industry`, `forecast`.

### Report Validity

Estimates whether a candidate is a structured report rather than a blog, landing page, or news snippet. Signals: document length, PDF format hints, report section headings, methodology and reference language.

### Quality

Six sub-components blended into a single score:

| Component | Weight | What it measures |
|---|---|---|
| methodology | 0.22 | Research design, survey description, benchmark process |
| reference | 0.22 | Citations, footnotes, named institutional sources |
| consistency | 0.18 | Coherent claims backed by data |
| structure | 0.14 | Report-like section organisation |
| data | 0.14 | Non-year numeric density |
| claim | 0.10 | Verifiable analytical claims |

### Authority

Classifies the publishing source (intergovernmental body, consulting firm, research provider, trade media, etc.).

## Verification and Reranking

Implemented in `source/verification.py`, `claim_verifier.py`, `citation_extractor.py`, and `verification_metrics.py`. The verifier:

- extracts important claims from top report text
- extracts URLs and numbered references
- maps claims to citation URLs
- checks numeric matches and token overlap
- classifies source reliability
- computes 12 verification metrics

Metrics are blended back into quality sub-components with bounded weights:

```text
methodology = 80% original + 20% verification
reference   = 65% original + 35% verification
consistency = 70% original + 30% verification
structure   = 85% original + 15% verification
data        = 70% original + 30% verification
claim       = 70% original + 30% verification
```

## Curated Benchmark

The benchmark lives in [`data/curated_benchmark/`](data/curated_benchmark/). It contains 20 queries, 5 candidate documents per query (100 total), relevance/ranking labels, and DEER-inspired quality labels.

Document URLs point to real, live reports found via web search — not synthetic placeholders.

### Refresh document URLs

Searches SerpApi for each query and replaces any stale URLs in `documents.csv`:

```bash
SERPAPI_API_KEY=<key> python scripts/update_document_urls.py

# Preview without writing
SERPAPI_API_KEY=<key> python scripts/update_document_urls.py --dry-run
```

### Re-annotate with Qwen

Annotates all 100 documents with Qwen-generated quality labels, using a hybrid strategy: Qwen scores quality dimensions (evidence, transparency, recency, authority, relevance), heuristics resolve structural fields (`result_class`, `ranking_preference`, `deer_method_label`) that the small model cannot reliably classify from short snippets.

```bash
# Requires USE_LOCAL_QWEN=1 and a downloaded Qwen2.5-*-Instruct model.
# Use a single venv Python to avoid cross-venv torchvision conflicts.
USE_LOCAL_QWEN=1 /path/to/venv/Scripts/python.exe scripts/annotate_benchmark.py

# Single query
USE_LOCAL_QWEN=1 /path/to/venv/Scripts/python.exe scripts/annotate_benchmark.py --query q003

# Preview without saving
USE_LOCAL_QWEN=1 /path/to/venv/Scripts/python.exe scripts/annotate_benchmark.py --dry-run

# Without Qwen (heuristic-only fallback)
python scripts/annotate_benchmark.py
```

### Benchmark evaluation commands

```powershell
# Validate and write benchmark starter
.\.venv\Scripts\python.exe source\runtime\curated_benchmark.py --write

# Rank benchmark documents through the scoring path
.\.venv\Scripts\python.exe source\runtime\curated_benchmark.py --rank

# Tune final score weights
.\.venv\Scripts\python.exe source\runtime\curated_benchmark.py --tune --top-n 10
```

## Local Qwen Integration

`local_qwen.py` provides optional on-device inference using a Hugging Face `Qwen2.5-*-Instruct` model. It is used in two places:

- **`assess_text_signals`** — scores reference quality, methodology, consistency, and source authority from fetched report text, feeding additional signal into the main scoring pipeline.
- **`suggest_search_queries`** / **`rewrite_search_query`** — generates improved search queries for a given topic.

Enable with:

```bash
USE_LOCAL_QWEN=1 python source/main.py "..."
```

The module discovers models and compatible venvs automatically. If inference fails, all functions return safe defaults and the pipeline continues without disruption.

## Design Notes

- **Search is swappable** without touching scoring or verification.
- **Scoring is weighted and interpretable** — each component has a documented formula and can be tuned independently.
- **Verification is deterministic** — no LLM, no external calls beyond what was already fetched.
- **Reflection and replanning** are controller-level behaviors separate from scoring logic.
- **The benchmark is small but real** — 20 curated queries, live report URLs, document-specific Qwen annotations.
