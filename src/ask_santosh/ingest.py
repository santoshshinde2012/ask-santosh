"""Ingestion pipeline: load documents -> chunk -> embed -> Chunk records.

Depends only on abstractions (DocumentSource, Chunker, Embedder). Persistence is
the caller's concern (a VectorStore), keeping this class single-purpose.
"""
from __future__ import annotations

import sys

from .models import Chunk, Document
from .protocols import Chunker, DocumentSource, Embedder


class IngestionPipeline:
    def __init__(
        self,
        source: DocumentSource,
        chunker: Chunker,
        embedder: Embedder,
        batch_size: int = 64,
    ) -> None:
        self._source = source
        self._chunker = chunker
        self._embedder = embedder
        self._batch = batch_size

    def build(self) -> list[Chunk]:
        chunks = [c for doc in self._source.load() for c in self._to_chunks(doc)]
        for start in range(0, len(chunks), self._batch):
            window = chunks[start : start + self._batch]
            vectors = self._embedder.embed([c.text for c in window])
            for chunk, vector in zip(window, vectors, strict=True):
                chunk.embedding = vector
        return chunks

    def _to_chunks(self, doc: Document) -> list[Chunk]:
        return [
            Chunk(id=f"{doc.url}#{i}", title=doc.title, url=doc.url, text=passage)
            for i, passage in enumerate(self._chunker.split(doc.text))
        ]


def main() -> int:
    """CLI entry point: build the index and persist it."""
    from .container import build_ingestion_pipeline, build_store_for_write
    from .settings import Settings

    settings = Settings.from_env()
    chunks = build_ingestion_pipeline(settings).build()
    if not chunks:
        print("No content ingested — check FEEDS.", file=sys.stderr)
        return 1
    store = build_store_for_write()
    store.upsert(chunks)
    store.save(settings.index_path)
    print(f"indexed {len(chunks)} chunks -> {settings.index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
