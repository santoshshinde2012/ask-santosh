"""Immutable domain models. No behavior, no I/O — just data."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Document:
    """A source article to be indexed."""

    title: str
    url: str
    text: str


@dataclass(slots=True)
class Chunk:
    """A retrievable passage derived from a Document, with its embedding."""

    id: str
    title: str
    url: str
    text: str
    embedding: list[float] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Source:
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class Answer:
    """The result of a RAG query."""

    text: str
    sources: tuple[Source, ...]
    retrieval_context: tuple[str, ...]
