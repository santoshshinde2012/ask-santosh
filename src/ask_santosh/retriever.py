"""Retrieval: embed the query, then ask the store for nearest chunks.

Depends only on the Embedder and VectorStore abstractions (Dependency
Inversion) — it never imports OpenAI or a concrete store.
"""
from __future__ import annotations

from .models import Chunk
from .protocols import Embedder, VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int = 4) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._embedder = embedder
        self._store = store
        self._top_k = top_k

    def retrieve(self, question: str, k: int | None = None) -> list[Chunk]:
        vector = self._embedder.embed([question])[0]
        return self._store.search(vector, k or self._top_k)
