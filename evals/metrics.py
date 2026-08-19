"""Compute retrieval metrics. The single source of truth for gate and badge.

`test_retrieval.py` asserts on these numbers and `report.py` publishes them, so
the badge can never claim something the gate did not actually measure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from corpus import DOCUMENTS, LABELLED
from local_embedding import HashingEmbedder

from ask_santosh.chunking import SlidingWindowChunker
from ask_santosh.ingest import IngestionPipeline
from ask_santosh.models import Document
from ask_santosh.retriever import Retriever
from ask_santosh.store import JSONVectorStore


@dataclass(frozen=True, slots=True)
class Metrics:
    recall_at_1: float
    recall_at_3: float
    mrr: float
    queries: int
    chunks: int
    misses: tuple[str, ...] = field(default_factory=tuple)


class _FixedSource:
    def load(self) -> list[Document]:
        return list(DOCUMENTS)


def build_store() -> JSONVectorStore:
    pipeline = IngestionPipeline(
        source=_FixedSource(),
        chunker=SlidingWindowChunker(900, 150),
        embedder=HashingEmbedder(),
    )
    store = JSONVectorStore()
    store.upsert(pipeline.build())
    return store


def compute(store: JSONVectorStore | None = None) -> Metrics:
    store = store or build_store()
    hits_1 = 0
    hits_3 = 0
    reciprocal = 0.0
    misses: list[str] = []

    for question, gold in LABELLED.items():
        ranked = [c.url for c in Retriever(HashingEmbedder(), store, top_k=5).retrieve(question)]
        if ranked and ranked[0] == gold:
            hits_1 += 1
        if gold in ranked[:3]:
            hits_3 += 1
        else:
            misses.append(question)
        if gold in ranked:
            reciprocal += 1.0 / (ranked.index(gold) + 1)

    n = len(LABELLED)
    return Metrics(
        recall_at_1=hits_1 / n,
        recall_at_3=hits_3 / n,
        mrr=reciprocal / n,
        queries=n,
        chunks=len(store),
        misses=tuple(misses),
    )
