"""RSS/Atom document source. Implements DocumentSource.

`feedparser` is imported lazily so the core package (and its unit tests) need no
third-party dependency to import.
"""
from __future__ import annotations

from .models import Document
from .textutils import clean_html


class RssDocumentSource:
    def __init__(self, feeds: list[str], min_chars: int = 200) -> None:
        if not feeds:
            raise ValueError("at least one feed URL is required")
        self._feeds = feeds
        self._min_chars = min_chars

    def load(self) -> list[Document]:
        import feedparser  # lazy: keeps the core import-light

        docs: list[Document] = []
        for url in self._feeds:
            parsed = feedparser.parse(url)
            for entry in parsed.entries:
                body = ""
                if entry.get("content"):
                    body = entry["content"][0].get("value", "")
                text = clean_html(body or entry.get("summary", ""))
                if len(text) < self._min_chars:
                    continue
                docs.append(
                    Document(
                        title=entry.get("title", "Untitled"),
                        url=entry.get("link", ""),
                        text=text,
                    )
                )
        return docs
