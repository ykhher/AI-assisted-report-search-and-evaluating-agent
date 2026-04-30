"""Document fetching and parsing utilities."""

from source.fetching.document_fetcher import fetch_document
from source.fetching.parser import parse_search_results
from source.fetching.text_parser import parse_report_text

__all__ = ["fetch_document", "parse_report_text", "parse_search_results"]
