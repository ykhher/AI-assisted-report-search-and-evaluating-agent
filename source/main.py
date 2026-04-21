"""Command-line entrypoint for the report discovery and evaluation agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow `python source/main.py` to work by ensuring the project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.controller import DEFAULT_MAX_STEPS, run_agent


try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for demos and local testing."""
    parser = argparse.ArgumentParser(
        description="Run the report discovery/evaluation agent from the command line.",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="User query for report discovery, for example: enterprise AI adoption report 2025",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum controller steps to allow before stopping.",
    )
    parser.add_argument(
        "--verify-top-n",
        type=int,
        default=3,
        help="Run lightweight verification notes on the top N ranked reports.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print controller plan/state details after the ranked summary.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full structured result as JSON instead of the readable summary.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide controller step logs while the agent runs.",
    )
    return parser.parse_args()


def _format_score(value: Any) -> str:
    """Format a numeric score for presentation."""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _short_summary(item: dict[str, Any]) -> str:
    """Build a short description of the report's main content."""
    title = str(item.get("title", "")).strip()
    report_type = str(item.get("report_type", "report")).strip() or "report"
    source = str(item.get("source", "")).strip()

    cleaned_title = title
    if cleaned_title.lower().endswith(" report"):
        cleaned_title = cleaned_title[:-7].strip()

    if cleaned_title:
        summary = f"This {report_type} covers {cleaned_title.lower()}."
    else:
        summary = f"This is a {report_type} document."

    if source:
        summary += f" It is published by {source}."

    verification_notes = item.get("verification_notes", [])
    if isinstance(verification_notes, list) and verification_notes:
        claim = str(verification_notes[0].get("claim", "")).strip()
        if claim:
            short_claim = claim.rstrip(".")
            summary += f" It highlights that {short_claim.lower()}."

    return summary


def _print_ranked_summary(result: dict[str, Any]) -> None:
    """Print a compact, presentation-friendly ranked summary."""
    ranked_results = result.get("ranked_results", [])
    stop_reason = result.get("stop_reason") or "completed"
    processing_time_ms = result.get("processing_time_ms", 0.0)

    print(f"\nQuery: {result.get('query', '')}")
    print(f"Stop reason: {stop_reason}")
    print(f"Processing time: {_format_score(processing_time_ms)} ms")

    if not ranked_results:
        print("\nNo ranked reports found.")
        return

    print("\nTop Ranked Reports")
    print("-" * 80)

    for index, item in enumerate(ranked_results, start=1):
        title = str(item.get("title", "Untitled report"))
        source = str(item.get("source") or item.get("url", ""))
        final_score = _format_score(item.get("score", item.get("final_score")))
        report_type = str(item.get("report_type", "unknown"))
        reason = str(item.get("reason", "")).strip()
        summary = _short_summary(item)

        print(f"{index}. {title}")
        print(f"   score={final_score}  type={report_type}  source={source}")
        if summary:
            print(f"   summary: {summary}")
        if reason:
            print(f"   why: {reason}")

        verification_notes = item.get("verification_notes", [])
        if isinstance(verification_notes, list) and verification_notes:
            top_note = verification_notes[0]
            claim = str(top_note.get("claim", "")).strip()
            confidence = str(top_note.get("confidence", "unknown")).strip()
            if claim:
                print(f"   verify: {confidence} confidence - {claim}")
        print()


def main() -> None:
    """Run the controller-backed CLI for report discovery and evaluation."""
    args = _parse_args()

    query = " ".join(args.query).strip()
    if not query:
        query = input("Enter query: ").strip()

    if not query:
        print("No query provided.")
        return

    result = run_agent(
        user_query=query,
        max_steps=max(1, args.max_steps),
        return_state=True,
        verbose=not args.quiet,
        verify_top_n=max(0, args.verify_top_n),
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    _print_ranked_summary(result)

    if args.debug:
        debug_payload = {
            "plan": result.get("plan", {}),
            "stop_reason": result.get("stop_reason"),
            "processing_time_ms": result.get("processing_time_ms"),
            "state": result.get("state", {}),
        }
        print("\nDebug Details")
        print("-" * 80)
        print(json.dumps(debug_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
