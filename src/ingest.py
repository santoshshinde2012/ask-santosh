"""Ingest: fetch the configured feeds, chunk each article, embed the chunks,
and persist a flat JSON vector index. Run once (or on a schedule) before serving.

    python -m src.ingest
"""
from __future__ import annotations

import html
import json
import re
import sys

import feedparser

from . import config

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _clean(raw: str) -> str:
    text = _TAG.sub(" ", raw or "")
    text = html.unescape(text)
    return _WS.sub(" ", text).strip()


def _chunk(text: str) -> list[str]:
    step = max(1, config.CHUNK_CHARS - config.CHUNK_OVERLAP)
    return [text[i : i + config.CHUNK_CHARS] for i in range(0, len(text), step)]


def _embed(texts: list[str]) -> list[list[float]]:
    resp = config.client().embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def build() -> int:
    records: list[dict] = []
    for feed_url in config.FEEDS:
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries:
            body = ""
            if entry.get("content"):
                body = entry["content"][0].get("value", "")
            body = body or entry.get("summary", "")
            text = _clean(body)
            if len(text) < 200:
                continue
            for j, chunk in enumerate(_chunk(text)):
                records.append(
                    {
                        "id": f"{entry.get('id', entry.get('link'))}#{j}",
                        "title": entry.get("title", "Untitled"),
                        "url": entry.get("link", ""),
                        "text": chunk,
                    }
                )

    if not records:
        print("No content ingested — check FEEDS.", file=sys.stderr)
        return 0

    # embed in batches to stay under request limits
    for start in range(0, len(records), 64):
        batch = records[start : start + 64]
        for rec, vec in zip(batch, _embed([r["text"] for r in batch])):
            rec["embedding"] = vec
        print(f"embedded {min(start + 64, len(records))}/{len(records)}")

    config.INDEX_PATH.parent.mkdir(exist_ok=True)
    config.INDEX_PATH.write_text(json.dumps(records))
    print(f"wrote {len(records)} chunks -> {config.INDEX_PATH}")
    return len(records)


if __name__ == "__main__":
    build()
