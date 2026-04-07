"""Entry point for the agent-based report discovery pipeline."""

from __future__ import annotations

import sys

from agent import agent_pipeline


if __name__ == "__main__":
    query = input("Enter query: ").strip()
    results = agent_pipeline(query)
    for r in results:
        print(r)
