"""Entry point for the agent-based report discovery pipeline."""

from __future__ import annotations

import json

from agent import agent_pipeline


if __name__ == "__main__":
    query = input("Enter query: ").strip()
    results = agent_pipeline(query, top_k=10)
    print(json.dumps(results, indent=2, ensure_ascii=False))
