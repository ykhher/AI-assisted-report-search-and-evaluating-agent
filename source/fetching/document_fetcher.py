"""Fetch documents and extract text from HTML or PDF content."""

from __future__ import annotations

import io
import re
from html.parser import HTMLParser
from typing import Any, Optional
from urllib.parse import urlparse

import requests


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0 Safari/537.36"
    )
}


class _HTMLTextParser(HTMLParser):
    """Lightweight HTML-to-text parser."""

    def __init__(self) -> None:
        super().__init__()
        self._in_script = False
        self._in_style = False
        self._title_parts: list[str] = []
        self._in_title = False
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        if tag_lower == "script":
            self._in_script = True
        elif tag_lower == "style":
            self._in_style = True
        elif tag_lower == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower == "script":
            self._in_script = False
        elif tag_lower == "style":
            self._in_style = False
        elif tag_lower == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_script or self._in_style:
            return

        value = data.strip()
        if not value:
            return

        if self._in_title:
            self._title_parts.append(value)
        else:
            self._text_parts.append(value)

    @property
    def title(self) -> str:
        """Return parsed title text."""
        return " ".join(self._title_parts).strip()

    @property
    def text(self) -> str:
        """Return parsed body text."""
        joined = "\n".join(self._text_parts)
        # Collapse excessive blank lines while preserving paragraph breaks.
        return re.sub(r"\n{3,}", "\n\n", joined).strip()


def detect_content_type(url: str, headers: dict[str, str] | None, content: bytes | None = None) -> str:
    """Detect content type as pdf, html, or unknown."""
    header_value = (headers or {}).get("Content-Type", "").lower()

    if "application/pdf" in header_value:
        return "pdf"
    if "text/html" in header_value or "application/xhtml+xml" in header_value:
        return "html"

    if content and content.startswith(b"%PDF"):
        return "pdf"

    parsed = urlparse(url)
    path_lower = parsed.path.lower()
    if path_lower.endswith(".pdf"):
        return "pdf"
    if path_lower.endswith(".html") or path_lower.endswith(".htm"):
        return "html"

    return "unknown"


def fetch_html_text(content: bytes, encoding_hint: str | None = None) -> dict[str, str]:
    """Extract title and text from HTML bytes."""
    encoding = encoding_hint or "utf-8"
    html = content.decode(encoding, errors="replace")

    parser = _HTMLTextParser()
    parser.feed(html)
    parser.close()

    return {
        "title": parser.title,
        "raw_text": parser.text,
    }


def _base_result(url: str) -> dict[str, Any]:
    """Build the standard fetch result shape."""
    return {
        "url": url,
        "status": "failed",
        "content_type": "unknown",
        "title": "",
        "raw_text": "",
        "error": "",
    }


def fetch_pdf_text(content: bytes) -> dict[str, str]:
    """Extract text from PDF bytes using pypdf when available."""
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return {
            "title": "",
            "raw_text": "",
            "error": "pypdf not installed; install with: pip install pypdf",
        }

    try:
        reader = PdfReader(io.BytesIO(content))
        pages_text: list[str] = []

        for page in reader.pages:
            text = page.extract_text() or ""
            cleaned = text.strip()
            if cleaned:
                pages_text.append(cleaned)

        title = ""
        if getattr(reader, "metadata", None) and reader.metadata:
            raw_title = reader.metadata.get("/Title")  # type: ignore[arg-type]
            title = str(raw_title).strip() if raw_title else ""

        return {
            "title": title,
            "raw_text": "\n\n".join(pages_text).strip(),
            "error": "",
        }
    except Exception as exc:
        return {
            "title": "",
            "raw_text": "",
            "error": f"pdf_parse_error: {exc}",
        }


def fetch_document(url: str, timeout: int = 15) -> dict[str, Any]:
    """Fetch a URL and return normalized text extraction output."""
    result = _base_result(url)

    if not isinstance(url, str) or not url.strip():
        result["error"] = "invalid_url"
        return result

    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers=DEFAULT_HEADERS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        result["error"] = f"request_error: {exc}"
        return result

    content = response.content or b""
    content_type = detect_content_type(url, dict(response.headers), content)
    result["content_type"] = content_type

    if content_type == "html":
        parsed = fetch_html_text(content, response.encoding)
        result["title"] = parsed.get("title", "")
        result["raw_text"] = parsed.get("raw_text", "")
        result["status"] = "ok"
        return result

    if content_type == "pdf":
        parsed = fetch_pdf_text(content)
        result["title"] = parsed.get("title", "")
        result["raw_text"] = parsed.get("raw_text", "")
        result["error"] = parsed.get("error", "")
        result["status"] = "ok"
        return result

    # Keep a best-effort text fallback for unknown content types.
    try:
        fallback_text = content.decode(response.encoding or "utf-8", errors="replace").strip()
    except Exception:
        fallback_text = ""

    result["status"] = "ok"
    result["raw_text"] = fallback_text
    result["error"] = "unknown_content_type"
    return result
