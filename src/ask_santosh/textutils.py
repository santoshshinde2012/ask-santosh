"""Small, pure text helpers (unit-testable, no dependencies)."""
from __future__ import annotations

import html
import re

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_html(raw: str) -> str:
    """Strip tags, unescape entities, and collapse whitespace."""
    text = _TAG.sub(" ", raw or "")
    text = html.unescape(text)
    return _WS.sub(" ", text).strip()
