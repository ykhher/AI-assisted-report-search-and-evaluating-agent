"""Helpers to export ranking results to CSV and JSON."""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# Default CSV columns for compact, readable export.
DEFAULT_CSV_FIELDS = [
    'index',
    'title',
    'url',
    'year',
    'source',
    'source_class',
    'report_type',
    'relevance_score',
    'report_validity_score',
    'quality_score',
    'authority_score',
    'final_score',
    'reasoning',
    'warnings',
]

# Extended CSV fields with additional classifier fields.
EXTENDED_CSV_FIELDS = DEFAULT_CSV_FIELDS + [
    'authority_prior',
    'report_validity_score_classifier',
]

# Minimal CSV fields.
MINIMAL_CSV_FIELDS = [
    'index',
    'title',
    'url',
    'source',
    'final_score',
    'reasoning',
]


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a result object into a plain dict when possible."""
    if isinstance(obj, dict):
        return obj
    
    if hasattr(obj, 'to_dict') and callable(obj.to_dict):
        try:
            return obj.to_dict()
        except Exception:
            pass
    
    try:
        from dataclasses import asdict
        return asdict(obj)
    except Exception:
        pass
    
    return obj


def _get_field_value(obj: Any, field: str, default: str = "") -> Any:
    """Read a field value, including dotted paths like score_breakdown.final_score."""
    result_dict = _to_dict(obj)
    
    if '.' in field:
        parts = field.split('.')
        value = result_dict
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value
    
    return result_dict.get(field, default)


def _format_field_for_csv(value: Any, max_length: int = 500) -> str:
    """Format one value for CSV output."""
    if value is None:
        return ""
    
    if isinstance(value, bool):
        return "yes" if value else "no"
    
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    
    if isinstance(value, float):
        return f"{value:.3f}"
    
    str_value = str(value)
    if len(str_value) > max_length:
        return str_value[:max_length] + "..."
    
    return str_value


def export_to_csv(
    results: Union[List[Dict], List[Any]],
    filepath: Union[str, Path],
    fields: Optional[List[str]] = None,
    include_header: bool = True,
    include_timestamp: bool = True,
) -> None:
    """Write ranked results to CSV."""
    if not results:
        raise ValueError("Results list is empty; nothing to export")
    
    filepath = Path(filepath)
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if fields is None:
        fields = DEFAULT_CSV_FIELDS
    
    first_dict = _to_dict(results[0])
    available_fields = set(first_dict.keys())
    if 'score_breakdown' in available_fields and isinstance(first_dict.get('score_breakdown'), dict):
        available_fields.update(first_dict['score_breakdown'].keys())
    
    fieldnames = [f for f in fields if f in available_fields or '.' in f]
    
    if not fieldnames:
        raise ValueError(f"No matching fields found. Available: {sorted(available_fields)}")
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if include_timestamp:
                f.write(f"# Exported: {datetime.now().isoformat()}\n")
            
            writer = csv.writer(f)
            if include_header:
                writer.writerow(fieldnames)
            for result in results:
                row = [
                    _format_field_for_csv(_get_field_value(result, field, default=""))
                    for field in fieldnames
                ]
                writer.writerow(row)
        
        print(f"[exporter] Exported {len(results)} result(s) to CSV: {filepath}")
    
    except IOError as exc:
        raise IOError(f"Failed to write CSV to {filepath}: {exc}")


def export_to_json(
    results: Union[List[Dict], List[Any], Any],
    filepath: Union[str, Path],
    indent: int = 2,
    include_metadata: bool = True,
) -> None:
    """Write ranked results to JSON."""
    filepath = Path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        if hasattr(results, 'to_dict') and hasattr(results, 'results'):
            data = results.to_dict()
        elif isinstance(results, list):
            data = [_to_dict(r) for r in results]
        else:
            data = _to_dict(results)

        if include_metadata and isinstance(data, list):
            data = {
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'result_count': len(data),
                    'format_version': '1.0',
                },
                'results': data,
            }
        elif include_metadata and isinstance(data, dict) and 'results' not in data:
            data = {
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'result_count': 1,
                    'format_version': '1.0',
                },
                'result': data,
            }

    except Exception as exc:
        raise ValueError(f"Cannot serialize results to JSON: {exc}")

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        
        result_count = data.get('result_count', len(data) if isinstance(data, list) else 1)
        print(f"[exporter] Exported {result_count} result(s) to JSON: {filepath}")

    except IOError as exc:
        raise IOError(f"Failed to write JSON to {filepath}: {exc}")


def export_batch(
    results: Union[List[Dict], List[Any], Any],
    output_dir: Union[str, Path] = ".",
    base_name: str = "results",
    formats: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Export results to one or more formats in a single call."""
    if formats is None:
        formats = ['csv', 'json']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    if 'csv' in formats:
        csv_path = output_dir / f"{base_name}.csv"
        try:
            export_to_csv(results, csv_path)
            paths['csv'] = csv_path
        except Exception as exc:
            print(f"[exporter] Warning: CSV export failed: {exc}")
    
    if 'json' in formats:
        json_path = output_dir / f"{base_name}.json"
        try:
            export_to_json(results, json_path)
            paths['json'] = json_path
        except Exception as exc:
            print(f"[exporter] Warning: JSON export failed: {exc}")
    
    return paths


# ============================================================================
# TEST HARNESS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("EXPORTER TEST")
    print("=" * 80)
    
    # Create sample results
    sample_results = [
        {
            'index': 1,
            'title': 'Global AI Market Report 2024',
            'url': 'https://example.com/ai-market',
            'year': 2024,
            'source': 'mckinsey.com',
            'source_class': 'consulting',
            'report_type': 'report',
            'relevance_score': 0.85,
            'report_validity_score': 0.78,
            'quality_score': 0.72,
            'authority_score': 0.88,
            'final_score': 0.79,
            'reasoning': 'High authority consulting firm with strong report structure',
            'warnings': [],
        },
        {
            'index': 2,
            'title': 'Cloud Computing Trends 2024',
            'url': 'https://research.example.org/cloud',
            'year': 2024,
            'source': 'research.example.org',
            'source_class': 'research_institute',
            'report_type': 'whitepaper',
            'relevance_score': 0.72,
            'report_validity_score': 0.68,
            'quality_score': 0.65,
            'authority_score': 0.75,
            'final_score': 0.70,
            'reasoning': 'Academic source with adequate documentation',
            'warnings': ['publication_date_inferred'],
        },
        {
            'index': 3,
            'title': 'My AI Tech Blog',
            'url': 'https://myblog.com/ai-thoughts',
            'year': None,
            'source': 'myblog.com',
            'source_class': 'blog',
            'report_type': 'blog',
            'relevance_score': 0.55,
            'report_validity_score': 0.35,
            'quality_score': 0.42,
            'authority_score': 0.30,
            'final_score': 0.45,
            'reasoning': 'Lower authority source, limited analytical depth',
            'warnings': ['low_confidence', 'no_publication_date'],
        },
    ]
    
    # Test 1: CSV export with default fields
    print("\n1. CSV Export (default fields)")
    try:
        export_to_csv(sample_results, 'test_results.csv')
        print("   ✓ Created test_results.csv")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: CSV export with custom fields
    print("\n2. CSV Export (custom fields)")
    try:
        export_to_csv(
            sample_results,
            'test_results_compact.csv',
            fields=MINIMAL_CSV_FIELDS,
        )
        print("   ✓ Created test_results_compact.csv")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: JSON export
    print("\n3. JSON Export")
    try:
        export_to_json(sample_results, 'test_results.json', indent=2)
        print("   ✓ Created test_results.json")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 4: Batch export
    print("\n4. Batch Export (CSV + JSON)")
    try:
        paths = export_batch(sample_results, output_dir='.', base_name='test_batch')
        for fmt, path in paths.items():
            print(f"   ✓ Created {path.name}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 5: Show file contents
    print("\n5. File Contents Sample")
    if os.path.exists('test_results.csv'):
        with open('test_results.csv', 'r', encoding='utf-8') as f:
            lines = f.readlines()[:4]
            print("   test_results.csv (first 4 lines):")
            for line in lines:
                print(f"     {line.rstrip()}")
    
    if os.path.exists('test_results.json'):
        with open('test_results.json', 'r', encoding='utf-8') as f:
            content = f.read()
            preview = content[:200]
            print(f"   test_results.json (first 200 chars):")
            print(f"     {preview}...")
    
    # Cleanup
    print("\n6. Cleanup")
    for filename in ['test_results.csv', 'test_results_compact.csv', 'test_results.json', 
                     'test_batch.csv', 'test_batch.json']:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"   Removed {filename}")
    
    print("\n" + "=" * 80)
    print("✓ All exporter tests completed")
    print("=" * 80)
