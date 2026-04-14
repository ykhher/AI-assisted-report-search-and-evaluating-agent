"""Compatibility wrappers for the older pipeline entrypoints.

The project now runs through the controller-backed agent path in `source.agent`.
This module stays around so older imports like `run_pipeline(...)` keep working
without carrying a second orchestration implementation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Allow direct execution with `python source/pipeline.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.agent import agent_pipeline


def run_pipeline(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Run the current agent path and return a ranked list."""
    results = agent_pipeline(query, top_k=top_k, output_format="list", export_results=False)
    return [item for item in results if isinstance(item, dict)]


def pipeline(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Backward-compatible alias for `run_pipeline`."""
    return run_pipeline(query, top_k=top_k)


def print_results(results: list[dict[str, Any]]) -> None:
    """Pretty-print ranked results to stdout as formatted JSON."""
    if not results:
        print("No valid reports found.")
        return

    print("\n" + "=" * 60)
    print("RANKED REPORT RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print("=" * 60 + "\n")
