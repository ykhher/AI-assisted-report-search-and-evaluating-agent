# Architecture Summary (Advisor Version)

## System Type
LLM-assisted retrieval and evaluation agent for industry report discovery.

## Core Goal
Identify credible industry reports from search results and rank them with transparent, explainable scoring.

## Runtime Flow
```text
Query Input
-> Query Rewrite (heuristics + local LLM)
-> API Search
-> Candidate Filter
-> Text/Metadata Signal Extraction
-> RQI Scoring + Relevance Blend
-> Ranked Top-K Output with Explanations
```

## Main Design Choices
- **Modular pipeline:** search, filtering, extraction, scoring, and ranking are split into separate modules.
- **API-first retrieval:** live search is prioritized for report discovery.
- **Hybrid scoring:** deterministic heuristics + local Qwen-assisted signal estimation.
- **Explainability-first output:** every ranked report includes a human-readable reason.
- **Graceful degradation:** if local LLM is unavailable/disabled, heuristic path still runs.

## In-Scope Capability
- report discovery
- non-report filtering
- quality signal extraction
- credibility ranking
- explanation generation

## Out-of-Scope Capability (Current Version)
- claim-level factual verification
- evidence graph construction
- legal-grade statement auditing

## MVP Definition
A working CLI agent that consistently returns ranked top-k report candidates in JSON format with `RQI`, `score`, and explanation fields, using the modular architecture above.
