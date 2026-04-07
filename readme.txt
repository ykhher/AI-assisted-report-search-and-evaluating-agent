Report Discovery and Credibility Ranking System
================================================

Overview
--------
This project is a local Python-based pipeline for discovering report-like documents,
extracting credibility signals, scoring them, and returning ranked results.

Current pipeline flow:
query -> expand_query -> search_reports -> filter_results/is_report ->
extract_signals -> compute_rqi -> final_score -> rank_reports -> output

Project Files
-------------
- main.py       : entry point for running the full pipeline
- pipeline.py   : end-to-end system integration
- search.py     : query expansion and mock/local search results
- filter.py     : coarse filtering helpers
- extractor.py  : Phase 3 signal extraction
- scoring.py    : Phase 4 scoring, ranking, and explanations
- API.py        : optional helper for HTTP search using requests
- dataset.json  : sample report metadata

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
3. Start the demo pipeline:
   python main.py

Example
-------
The entry point currently runs:
    run_pipeline("AI market report")

Expected output format:
{
  'title': '...',
  'url': '...',
  'RQI': 0.82,
  'score': 0.78,
  'reason': 'contains methodology, includes references'
}

Notes
-----
- The project is fully runnable locally.
- Search uses mock/local results by default.
- No external APIs are required for the main demo pipeline.
- `API.py` includes an optional requests-based HTTP helper.
