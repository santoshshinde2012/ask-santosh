"""In-memory vector store with JSON persistence.

Implements the VectorStore protocol. Kept intentionally simple (linear cosine
scan) — correct and dependency-free for a personal corpus. Because callers
depend only on the protocol, this can be replaced by pgvector/FAISS/Qdrant
later with no changes upstream (Open/Closed + Liskov).
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from .models import Chunk
from .similarity import cosine


class JSONVectorStore:
    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks: list[Chunk] = list(chunks or [])

    def __len__(self) -> int:
        return len(self._chunks)

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        by_id = {c.id: c for c in self._chunks}
        for c in chunks:
            by_id[c.id] = c
        self._chunks = list(by_id.values())

    def search(self, vector: Sequence[float], k: int) -> list[Chunk]:
        if k <= 0:
            raise ValueError("k must be positive")
        scored = sorted(
            self._chunks,
            key=lambda c: cosine(vector, c.embedding),
            reverse=True,
        )
        return scored[:k]

    # --- persistence ---------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"id": c.id, "title": c.title, "url": c.url, "text": c.text, "embedding": c.embedding}
            for c in self._chunks
        ]
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> JSONVectorStore:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run the ingestion step first.")
        raw = json.loads(path.read_text())
        return cls([Chunk(**rec) for rec in raw])
