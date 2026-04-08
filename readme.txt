LLM-Powered Report Credibility AI Agent
=======================================

Overview
--------
This project is a local Python-based search-and-evaluation AI agent for discovering
report-like documents, evaluating their credibility, and ranking them using both
rule-based signals and a locally hosted LLM.

The agent combines:
- API-based search for live retrieval in the main workflow
- heuristic report detection
- credibility signal extraction
- local Qwen-based LLM scoring
- final ranking and explanation generation

Main agent flow:
user query -> agent/query rewrite -> API search -> filter_results/is_report ->
extract_signals (heuristics + LLM) -> compute_rqi -> final_score -> rank_reports -> output

Project Files
-------------
- main.py         : entry point for running the main API-based AI agent
- agent.py        : iterative agent loop and search orchestration
- local_qwen.py   : local Qwen model loading, prompting, and LLM-based scoring
- pipeline.py     : end-to-end system integration
- search.py       : query expansion, API retrieval, and fallback search logic
- filter.py       : coarse filtering helpers
- extractor.py    : credibility signal extraction (heuristics + LLM)
- scoring.py      : final RQI scoring, ranking, and explanations
- API.py          : HTTP search client used by the main workflow
- dataset.json    : sample report metadata
- test_dataset_scores.py : dataset-level scoring utility

Requirements
------------
- Python 3.10+ recommended
- Install dependencies with:
  pip install -r requirements.txt

How to Run
----------
1. Open a terminal in this project folder.
2. Install dependencies:
   pip install -r requirements.txt
3. Start the main AI agent (uses API search first and returns the top 10 ranked reports):
   python main.py
4. To score the sample dataset directly:
   python test_dataset_scores.py

Scoring Metrics and Formula
---------------------------
The main evaluation score is the **Report Quality Index (RQI)**.

Core metrics used by the agent:
- `methodology`        : evidence of survey design, sampling, research process
- `citation_score`     : references, footnotes, source notes, institutional citations
- `source`             : credibility of the named publisher or organization
- `consistency_score`  : whether findings, numbers, and conclusions align coherently
- `data_density`       : amount of quantitative evidence in the text
- `structure_score`    : presence of report-like sections
- `recency`            : preference for more recent material
- `claim_density`      : analytical or forecast-style wording

Main RQI formula:
    RQI =
        0.12 * methodology +
        0.20 * citation_score +
        0.15 * data_component +
        0.14 * source +
        0.10 * recency +
        0.12 * structure_score +
        0.07 * claim_density +
        0.10 * consistency_score

The final ranking score is then computed as a blend of:
- query relevance
- RQI credibility

Example output JSON (top 10 ranked reports):
[
  {
    "title": "Artificial Intelligence Index Report 2024",
    "url": "https://hai.stanford.edu/assets/files/hai_ai-index-report-2024-smaller2.pdf",
    "RQI": 0.84,
    "score": 0.79,
    "reason": "contains methodology, strong citation support, high-confidence methodology and citations"
  },
  {
    "title": "The State of AI in Early 2024",
    "url": "https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai-2024",
    "RQI": 0.81,
    "score": 0.76,
    "reason": "strong citation support, internally consistent analysis, recent publication"
  }
]

The main workflow returns the top 10 ranked reports when enough valid candidates are available.

Notes
-----
- The project is fully runnable locally.
- The main workflow is designed to use live API search for retrieval.
- `API.py` provides the search client used by the main agent flow.
- The agent is built as an LLM-assisted search and evaluation AI agent.
- A locally downloaded Qwen model is reused for:
  - search-query rewriting
  - reference detection
  - methodology detection
  - consistency scoring
  - source credibility estimation
- The ranking system blends heuristic signals with local LLM judgments for more realistic report evaluation.
- You can disable local LLM scoring by setting `USE_LOCAL_QWEN_SIGNALS=0` if you want a heuristic-only run.
