"""Retrieval quality gate that needs no API key, and therefore always runs.

The DeepEval suite in `test_rag.py` judges *answers*, needs a paid judge model,
and is skipped when no key is configured. That left the repository with no
quality gate on the majority of pushes. This suite closes that gap: it scores
*retrieval* against a fixed labelled corpus using a deterministic local
embedder, so it runs on every push, costs nothing, and cannot flake.

It exercises the real chunker, real store and real retriever — only the
embedding provider is swapped, which the Embedder protocol exists to allow.

Thresholds sit just under measured behaviour (recall@1 0.857, recall@3 1.000,
MRR 0.929). Because every input is fixed there is no variance to absorb, so a
drop below these means retrieval genuinely regressed.
"""
from __future__ import annotations

import pytest
from corpus import DOCUMENTS, LABELLED
from local_embedding import HashingEmbedder

from ask_santosh.chunking import SlidingWindowChunker
from ask_santosh.ingest import IngestionPipeline
from ask_santosh.retriever import Retriever
from ask_santosh.store import JSONVectorStore

MIN_RECALL_AT_1 = 0.80
MIN_RECALL_AT_3 = 1.00
MIN_MRR = 0.90


class _FixedSource:
    """DocumentSource over the labelled corpus."""

    def load(self) -> list:
        return list(DOCUMENTS)


@pytest.fixture(scope="module")
def store() -> JSONVectorStore:
    embedder = HashingEmbedder()
    pipeline = IngestionPipeline(
        source=_FixedSource(),
        chunker=SlidingWindowChunker(900, 150),
        embedder=embedder,
    )
    s = JSONVectorStore()
    s.upsert(pipeline.build())
    return s


def _ranked_urls(store: JSONVectorStore, question: str, k: int) -> list[str]:
    retriever = Retriever(HashingEmbedder(), store, top_k=k)
    return [c.url for c in retriever.retrieve(question)]


def test_index_is_populated(store: JSONVectorStore) -> None:
    assert len(store) == len(DOCUMENTS), "chunker or ingest dropped documents"


@pytest.mark.parametrize("question,expected", sorted(LABELLED.items()))
def test_gold_document_is_retrieved(store, question: str, expected: str) -> None:
    """Every labelled question must surface its source within the top 3."""
    assert expected in _ranked_urls(store, question, 3), (
        f"{expected} missing from top-3 for {question!r}"
    )


def test_recall_at_1(store: JSONVectorStore) -> None:
    hits = sum(
        _ranked_urls(store, q, 1)[0] == gold for q, gold in LABELLED.items()
    )
    recall = hits / len(LABELLED)
    assert recall >= MIN_RECALL_AT_1, f"recall@1 {recall:.3f} < {MIN_RECALL_AT_1}"


def test_recall_at_3(store: JSONVectorStore) -> None:
    hits = sum(gold in _ranked_urls(store, q, 3) for q, gold in LABELLED.items())
    recall = hits / len(LABELLED)
    assert recall >= MIN_RECALL_AT_3, f"recall@3 {recall:.3f} < {MIN_RECALL_AT_3}"


def test_mean_reciprocal_rank(store: JSONVectorStore) -> None:
    total = 0.0
    for question, gold in LABELLED.items():
        ranked = _ranked_urls(store, question, 5)
        if gold in ranked:
            total += 1.0 / (ranked.index(gold) + 1)
    mrr = total / len(LABELLED)
    assert mrr >= MIN_MRR, f"MRR {mrr:.3f} < {MIN_MRR}"
