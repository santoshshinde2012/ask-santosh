"""Abstractions the rest of the system depends on (Dependency Inversion).

Every collaborator is a narrow, single-purpose Protocol (Interface Segregation),
so concrete providers — OpenAI, GitHub Models, a local model, an in-memory or a
pgvector store — are swappable without touching business logic (Open/Closed).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .models import Chunk, Document


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class ChatClient(Protocol):
    """Generates a completion from a system + user message."""

    def complete(self, system: str, user: str) -> str: ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a long text into passages."""

    def split(self, text: str) -> list[str]: ...


@runtime_checkable
class DocumentSource(Protocol):
    """Yields the documents to be indexed."""

    def load(self) -> list[Document]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Persists embedded chunks and answers nearest-neighbour queries."""

    def upsert(self, chunks: Sequence[Chunk]) -> None: ...

    def search(self, vector: Sequence[float], k: int) -> list[Chunk]: ...

    def __len__(self) -> int: ...
