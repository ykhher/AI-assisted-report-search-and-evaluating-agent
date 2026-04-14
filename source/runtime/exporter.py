"""Helpers to export ranking results to CSV and JSON."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_CSV_FIELDS = [
    "index",
    "title",
    "url",
    "year",
    "source",
    "source_class",
    "report_type",
    "relevance_score",
    "report_validity_score",
    "quality_score",
    "authority_score",
    "final_score",
    "reasoning",
    "warnings",
]

EXTENDED_CSV_FIELDS = DEFAULT_CSV_FIELDS + [
    "authority_prior",
    "report_validity_score_classifier",
]

MINIMAL_CSV_FIELDS = [
    "index",
    "title",
    "url",
    "source",
    "final_score",
    "reasoning",
]


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert a result object into a plain dictionary when possible."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            data = obj.to_dict()
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    if is_dataclass(obj):
        try:
            return asdict(obj)
        except Exception:
            pass
    return {}


def _get_field_value(obj: Any, field: str, default: str = "") -> Any:
    """Read a field value, including dotted paths like score_breakdown.final_score."""
    value: Any = _to_dict(obj)
    for part in field.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value


def _format_field_for_csv(value: Any, max_length: int = 500) -> str:
    """Format one value for CSV output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value)
    if isinstance(value, float):
        return f"{value:.3f}"

    text = str(value)
    return text if len(text) <= max_length else text[:max_length] + "..."


def export_to_csv(
    results: list[dict] | list[Any],
    filepath: str | Path,
    fields: list[str] | None = None,
    include_header: bool = True,
    include_timestamp: bool = True,
) -> None:
    """Write ranked results to CSV."""
    if not results:
        raise ValueError("Results list is empty; nothing to export")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    selected_fields = fields or DEFAULT_CSV_FIELDS
    first_dict = _to_dict(results[0])
    available_fields = set(first_dict.keys())
    breakdown = first_dict.get("score_breakdown")
    if isinstance(breakdown, dict):
        available_fields.update(breakdown.keys())

    fieldnames = [field for field in selected_fields if field in available_fields or "." in field]
    if not fieldnames:
        raise ValueError(f"No matching fields found. Available: {sorted(available_fields)}")

    try:
        with filepath.open("w", newline="", encoding="utf-8") as handle:
            if include_timestamp:
                handle.write(f"# Exported: {datetime.now().isoformat()}\n")

            writer = csv.writer(handle)
            if include_header:
                writer.writerow(fieldnames)

            for result in results:
                writer.writerow(
                    [_format_field_for_csv(_get_field_value(result, field, default="")) for field in fieldnames]
                )
    except OSError as exc:
        raise OSError(f"Failed to write CSV to {filepath}: {exc}") from exc

    print(f"[exporter] Exported {len(results)} result(s) to CSV: {filepath}")


def export_to_json(
    results: list[dict] | list[Any] | Any,
    filepath: str | Path,
    indent: int = 2,
    include_metadata: bool = True,
) -> None:
    """Write ranked results to JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(results, "to_dict") and hasattr(results, "results"):
        data: Any = results.to_dict()
    elif isinstance(results, list):
        data = [_to_dict(item) for item in results]
    else:
        data = _to_dict(results)

    if include_metadata and isinstance(data, list):
        result_count = len(data)
        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "result_count": result_count,
                "format_version": "1.0",
            },
            "results": data,
        }
    elif include_metadata and isinstance(data, dict) and "results" not in data:
        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "result_count": 1,
                "format_version": "1.0",
            },
            "result": data,
        }

    try:
        with filepath.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=indent, ensure_ascii=False, default=str)
    except OSError as exc:
        raise OSError(f"Failed to write JSON to {filepath}: {exc}") from exc

    result_count = 1
    if isinstance(data, dict):
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            result_count = int(metadata.get("result_count", 1))
    elif isinstance(data, list):
        result_count = len(data)

    print(f"[exporter] Exported {result_count} result(s) to JSON: {filepath}")


def export_batch(
    results: list[dict] | list[Any] | Any,
    output_dir: str | Path = ".",
    base_name: str = "results",
    formats: list[str] | None = None,
) -> dict[str, Path]:
    """Export results to one or more formats in a single call."""
    selected_formats = formats or ["csv", "json"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    if "csv" in selected_formats:
        csv_path = output_dir / f"{base_name}.csv"
        try:
            export_to_csv(results, csv_path)
            paths["csv"] = csv_path
        except Exception as exc:
            print(f"[exporter] Warning: CSV export failed: {exc}")

    if "json" in selected_formats:
        json_path = output_dir / f"{base_name}.json"
        try:
            export_to_json(results, json_path)
            paths["json"] = json_path
        except Exception as exc:
            print(f"[exporter] Warning: JSON export failed: {exc}")

    return paths
