# Industry Report Discovery Agent

This project is an agent for finding industry and research reports, filtering weak or non-report results, scoring report quality, and returning a ranked list with short explanations.

The code started as a retrieval-and-ranking pipeline and has moved toward an explicit agent loop with planning, tool use, reflection, replanning, verification, and post-verification reranking.

## What It Does

Given a query like `"enterprise AI adoption report 2025"`, the agent will:

- interpret the request into a structured plan
- search the web through SerpApi Google Search
- normalize and filter candidate documents
- fetch and parse fuller document text when possible
- classify source authority and document type
- extract credibility and analytical quality signals
- score and rank candidate reports
- reflect on result quality and replan if results are weak
- extract claims from top reports
- extract citations from report text
- run deterministic lightweight claim verification
- compute verification metrics
- recompute verification-adjusted quality
- rerank verified results when verification changes the evidence picture

The verification layer is intentionally lightweight. It does not use an LLM. It checks local report evidence, extracted citation URLs, numeric matches, source reliability, and claim/context overlap. This makes the behavior explainable and suitable for a capstone system.

## System Flow

```text
User query
-> planner
-> agent state
-> controller loop
-> search / fetch / parse / classify / score / rank
-> reflect
-> replan and repeat if quality is weak
-> verify top reports
-> recompute verification-adjusted quality
-> rerank
-> final ranked results
```

The controller is deterministic. Reflection diagnoses failures such as topic drift, weak quality signals, too few results, low authority, stale results, or fetch failures. The agent can replan and search again until quality is sufficient or the replan budget is exhausted.

## Main Files

```text
agent/
  main.py
  README.md
  requirements.txt
  local_qwen.py
  source/
    main.py
    agent.py
    controller.py
    planner.py
    agent_state.py
    tool_registry.py
    reflection.py
    verification.py
    verification_metrics.py
    claim_verifier.py
    citation_extractor.py
    search.py
    API.py
    filter.py
    extractor.py
    scoring.py
    query_handler.py
    pipeline.py
    classifier/
      source_classifier.py
      report_classifier.py
    fetching/
      document_fetcher.py
      text_parser.py
      parser.py
    runtime/
      curated_benchmark.py
      evaluate_agent.py
      iteration_controller.py
      exporter.py
      schemas.py
  data/
    dataset.json
    benchmark_queries.csv
    benchmark_labels.json
    curated_benchmark/
      queries.csv
      documents.csv
      retrieval_annotations.csv
      quality_annotations.csv
      README.md
```

## Search API

The live search wrapper is [source/API.py](source/API.py). It uses SerpApi Google Search:

```text
https://serpapi.com/search.json
```

The API key can be provided with:

```powershell
$env:SERPAPI_API_KEY="your-key"
```

Do not commit API keys to the repository. If `SERPAPI_API_KEY` is not set, live search will fail and the agent can fall back to local mock results where that path is enabled.

## Scoring

The ranker uses four interpretable components:

- `relevance_score`
- `report_validity_score`
- `quality_score`
- `authority_score`

These are computed in [source/scoring.py](source/scoring.py).

### Final Score

The current final ranking formula is:

```text
final_score =
  0.35 * relevance_score
+ 0.20 * report_validity_score
+ 0.30 * quality_score
+ 0.15 * authority_score
```

This balance is designed for live web retrieval, where many search results provide only snippets or PDF landing pages.

### Relevance

`relevance_score` measures query-topic overlap:

```text
relevance_score =
  overlap(query_terms, text_terms) / number_of_query_terms
```

Generic report-search terms such as `pdf`, `report`, `industry`, `analysis`, and `forecast` are ignored when possible so the score focuses on topic words.

### Report Validity

`report_validity_score` estimates whether a candidate is a real report rather than a weak page, blog, landing page, or generic article.

It uses:

- document length
- PDF/report-like format hints
- report sections
- methodology, references, or statistical language
- classifier validity from `report_classifier.py`

Formula:

```text
report_validity_score =
  0.35 * length_component
+ 0.25 * report_format_component
+ 0.20 * section_component
+ 0.20 * evidence_component
```

There is also snippet-aware fallback logic so short but clearly report-like SERP results are not crushed to near zero.

### Quality

`quality_score` measures analytical and verification-friendly report quality.

It is made from six components:

```text
quality_score =
  0.22 * methodology
+ 0.22 * reference
+ 0.18 * consistency
+ 0.14 * structure
+ 0.14 * data
+ 0.10 * claim
```

In code, `reference` is named `citation`, `data` is named `data_support`, and `claim` is named `claim_density`.

#### Methodology

Methodology asks whether the report explains how its findings were produced.

Text signals include:

- `methodology`
- `methods`
- `research design`
- `data collection`
- `sampling`
- `we conducted`
- `we analyzed`

After verification, methodology is also adjusted by:

- reproducibility
- reliability
- verified claims ratio

#### Reference

Reference asks whether the report cites sources and whether those citations support the claims.

Text signals include:

- footnotes
- numbered references
- reference sections
- source notes
- known institutions such as World Bank, OECD, IEA, IPCC, UN, McKinsey

Verification signals include:

- citation accuracy
- supported references per shown reference
- supported references per used reference
- used references per shown reference
- source reliability
- reference diversity HHI

#### Consistency

Consistency asks whether the report's claims are coherent and evidence-backed.

Text signals include:

- numbers inside analytical sections
- claim language such as `increase`, `decrease`, `forecast`, `projected`, and `expected`
- parsed statistical language

Verification signals include:

- claim accuracy
- external claim accuracy

#### Structure

Structure asks whether the document is organized like a real report.

Text signals include sections such as:

- introduction
- methodology
- results
- conclusion
- summary
- assumptions
- references

Verification signals include:

- used references per shown reference
- verified claims ratio

#### Data

Data asks whether the report uses concrete quantitative evidence.

The system counts non-year numeric tokens such as:

- percentages
- counts
- rates
- decimals

Calendar years like `2024` and `2025` are excluded from numeric density.

The raw data metric is:

```text
data_density =
  non_year_numeric_tokens / total_words
```

Then:

```text
data_support = data_density ^ 1.3
```

Verification adjusts this with external numeric claim accuracy.

#### Claim

Claim asks whether the report makes meaningful analytical claims and whether those claims are verifiable.

Text signals include claim words such as:

- increase
- decrease
- forecast
- projected
- expected

Verification signals include:

- verified claims ratio
- average citations per claim
- external claim accuracy

## Verification And Reranking

Verification is implemented in:

- [source/verification.py](source/verification.py)
- [source/claim_verifier.py](source/claim_verifier.py)
- [source/citation_extractor.py](source/citation_extractor.py)
- [source/verification_metrics.py](source/verification_metrics.py)

The verifier does not use an LLM. It:

- extracts important claims
- extracts URLs and numbered references from report text
- maps claims to citation URLs when possible
- checks numeric matches and token overlap
- classifies source reliability
- computes verification metrics

Verification metrics include:

- claim accuracy
- external claim accuracy
- external numeric claim accuracy
- citation accuracy
- supported references per shown reference
- supported references per used reference
- used references per shown reference
- reproducibility
- reliability
- reference diversity HHI
- verified claims ratio
- average citations per claim

These metrics are blended back into the six quality components. The blend is bounded:

```text
methodology = 80% original + 20% verification
reference   = 65% original + 35% verification
consistency = 70% original + 30% verification
structure   = 85% original + 15% verification
data        = 70% original + 30% verification
claim       = 70% original + 30% verification
```

After verification, the controller recomputes a verification-adjusted final score and reranks the verified reports. Verified results may include:

```text
verification_adjusted_quality_score
pre_verification_score
pre_verification_rank
verification_metrics
verified_claims
extracted_citations
```

## Curated Benchmark

The curated benchmark lives in [data/curated_benchmark](data/curated_benchmark).

It contains:

- `queries.csv`: 20 representative report-search queries
- `documents.csv`: 5 candidate documents per query
- `retrieval_annotations.csv`: relevance and ranking labels
- `quality_annotations.csv`: compressed DEER-inspired quality labels

Generate and validate the starter benchmark:

```powershell
.\.venv\Scripts\python.exe source\runtime\curated_benchmark.py --write
```

Rank benchmark documents through the existing scoring path:

```powershell
.\.venv\Scripts\python.exe source\runtime\curated_benchmark.py --rank
```

Tune final score weights:

```powershell
.\.venv\Scripts\python.exe source\runtime\curated_benchmark.py --tune --top-n 10
```

The benchmark uses DEER only as an expert-informed annotation framework. It does not copy the full DEER rubric.

## Installation

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

The project works without optional local Qwen support. If local Qwen is enabled through environment variables, it can provide extra signal estimates, but the main pipeline is deterministic and does not require it.

## Running The Agent

Readable CLI:

```powershell
.\.venv\Scripts\python.exe source\main.py "enterprise AI adoption benchmark report 2025"
```

With debug state:

```powershell
.\.venv\Scripts\python.exe source\main.py "enterprise AI adoption benchmark report 2025" --debug
```

With JSON output:

```powershell
.\.venv\Scripts\python.exe source\main.py "enterprise AI adoption benchmark report 2025" --json
```

Control verification depth:

```powershell
.\.venv\Scripts\python.exe source\main.py "enterprise AI adoption benchmark report 2025" --verify-top-n 3
```

## Example Output

Readable CLI output looks like this:

```text
Query: enterprise AI adoption benchmark report 2025
Stop reason: sufficient_quality
Processing time: 1842.317 ms

Top Ranked Reports
--------------------------------------------------------------------------------
1. State of Enterprise AI Adoption Report 2025
   score=0.781  type=report  source=https://example.com/report.pdf
   summary: This report covers state of enterprise ai adoption report 2025.
   why: contains methodology, strong citation support, internally consistent analysis
   verify: high confidence - Adoption increased by 35% in 2025.
```

## Design Notes

The system is intentionally modular:

- search can be swapped without changing scoring
- scoring is interpretable and weighted
- verification is deterministic and explainable
- reflection and replanning are controller-level behaviors
- the curated benchmark is small but defensible

This makes the project suitable for explaining, testing, and improving each stage independently.
