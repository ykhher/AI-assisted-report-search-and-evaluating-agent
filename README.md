# Industry Report Discovery Agent

This project is a Python agent for finding industry reports, filtering out weak or non-report results, scoring report quality, and returning a ranked list with short explanations.

The code started as a retrieval-and-ranking pipeline and has gradually moved toward a more explicit agent structure. Today, the system uses a simple controller loop that plans, searches, filters, fetches, scores, reflects, and can replan when results look weak.

It is built for practical report discovery and evaluation. It is not a full fact-checking system, and it should not be treated as one.

## What it does

Given a query like `"enterprise AI adoption report 2025"`, the agent will:

- interpret the request into a small structured plan
- search for candidate documents
- filter out pages that do not look report-like
- optionally fetch and parse fuller document text
- classify the source and report type
- extract quality and credibility signals
- score and rank the best candidates
- optionally attach lightweight verification notes to the top few results

The verification step is intentionally modest. It only adds internal evidence notes such as whether a claim looks measurable or appears methodology-dependent. It does not verify claims against outside sources.

## How the system is organized

The current flow looks like this:

```text
User query
-> planner
-> agent state
-> controller loop
-> tool selection
-> search / fetch / parse / classify / score / rank
-> reflection
-> optional replanning
-> final ranked results
```

The important thing to know is that the older pipeline logic still exists, but the newer controller-oriented path is now the main way the project runs.

## Step-by-step workflow

Step 1. User query -> structured task plan  
File: `planner.py`

Step 2. Agent controller initializes state and action loop  
File: `controller.py`

Step 3. Expand query into search-friendly variants  
Files: `query_handler.py`, `search.py`

Step 4. Retrieve candidate reports from API or fallback search  
File: `search.py`

Step 5. Normalize and coarsely filter raw search results  
Files: raw search parser, `filter.py`

Step 6. Fetch full document content for top candidates  
File: `document_fetcher.py`

Step 7. Parse document text into sections and lightweight stats  
File: `text_parser.py`

Step 8. Classify document type and source authority class  
Files: `report_classifier.py`, `source_classifier.py`

Step 9. Extract credibility and analytical quality signals  
File: `extractor.py`

Step 10. Compute interpretable sub-scores and final ranking score  
File: `scoring.py`

Step 11. Rank candidate reports  
File: `scoring.py`

Step 12. Run post-ranking claim extraction and lightweight verification  
File: `verification.py`

Step 13. Reflect on progress and diagnose failure modes  
File: `reflection.py`

Step 14. If needed, rewrite query and replan  
Files: `iteration_controller.py`, `controller.py`

Step 15. Return final ranked results  
Files: `agent.py`, `tool_registry.py`, `pipeline.py`

## Project layout

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
    search.py
    API.py
    filter.py
    extractor.py
    scoring.py
    query_handler.py
    pipeline.py
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
```

## Main modules

- `main.py`
  Repo-root entrypoint. This is the easiest way to run the project from the command line.

- `source/main.py`
  CLI wrapper around the controller-based agent path.

- `source/controller.py`
  The main control loop. It chooses actions, runs tools, updates state, reflects on progress, and replans when needed.

- `source/planner.py`
  Turns a raw query into a lightweight structured plan.

- `source/agent_state.py`
  Holds working memory for one run, including candidates, failed URLs, reflections, and action history.

- `source/tool_registry.py`
  Exposes the core retrieval, parsing, classification, scoring, and ranking functions as named tools.

- `source/reflection.py`
  Diagnoses weak progress and tells the controller whether it should stop or replan.

- `source/search.py`
  Handles live search plus a local fallback index when live retrieval is unavailable.

- `source/extractor.py`
  Extracts credibility and report-quality signals from text and metadata.

- `source/scoring.py`
  Computes component scores and the final rank score.

- `source/fetching/*`
  Fetches document content and parses simple structure from text.

- `source/classifier/*`
  Classifies source authority and document type.

- `source/runtime/evaluate_agent.py`
  Runs a lightweight benchmark against labeled query data.

## Scoring approach

The ranker uses four interpretable components:

- `relevance_score`
- `report_validity_score`
- `quality_score`
- `authority_score`

These are combined into the final score in `source/scoring.py`:

```text
final_score =
  0.40 * relevance_score +
  0.20 * report_validity_score +
  0.25 * quality_score +
  0.15 * authority_score
```

There is also a compatibility `RQI` field in some parts of the codebase, but the main ranking path now relies on the fuller structured score.

## Installation

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

### Search API

If you want live web retrieval, set one of these environment variables:

```bash
SEARCHAPI_API_KEY=...
```

or

```bash
YDC_API_KEY=...
```

If no API key is available, the project can still run using the local mock fallback inside `source/search.py`.

### Local Qwen

Local Qwen support is optional and is now off by default to keep runs faster. If you want to enable it, set `USE_LOCAL_QWEN=1`. If it is unavailable, the project still works with heuristic logic.

Useful environment variables:

- `USE_LOCAL_QWEN=1`
- `USE_LOCAL_QWEN_SIGNALS=0`
- `QWEN_MODEL_PATH=...`
- `QWEN_EXTRA_SITE_PACKAGES=...`

## Running the agent

The simplest way to run it:

```bash
python main.py "enterprise AI adoption report 2025"
```

You can also run it interactively:

```bash
python main.py
```

Useful options:

```bash
python main.py "enterprise AI adoption report 2025" --debug
python main.py "enterprise AI adoption report 2025" --json
python main.py "enterprise AI adoption report 2025" --quiet
python main.py "enterprise AI adoption report 2025" --verify-top-n 2
```

## Example output

Readable CLI output looks like this:

```text
Query: enterprise AI adoption report 2025
Stop reason: sufficient_quality
Processing time: 1842.317 ms

Top Ranked Reports
--------------------------------------------------------------------------------
1. Enterprise AI Adoption Benchmark 2025
   score=0.781  type=benchmark  source=mckinsey.com
   why: contains methodology, strong citation support, internally consistent analysis
   verify: medium confidence - Adoption is expected to reach 68% by 2025.
   url: https://example.com/enterprise-ai-benchmark-2025.pdf
```

If you pass `--json`, the CLI prints the fuller structured payload, including plan and state details when `--debug` is enabled.

## Benchmark evaluation

You can run the lightweight benchmark like this:

```bash
python -m source.runtime.evaluate_agent \
  --queries data/benchmark_queries.csv \
  --labels data/benchmark_labels.json \
  --output outputs/benchmark_results.json \
  --k 5
```

## What this project is not

This project is not:

- a full external-source fact-checking engine
- a legal or compliance review tool
- a general-purpose research assistant

It is best understood as a practical report discovery and ranking agent with explainable heuristics and a small amount of optional local-model support.

## Notes

- The controller-first path is the main execution path now.
- The older pipeline entrypoints are still kept around for compatibility.
- Some modules still carry compatibility wrappers from earlier versions of the project. That is deliberate and helps keep the system stable while the architecture evolves.
