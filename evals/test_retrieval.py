"""Retrieval quality gate that needs no API key, and therefore always runs.

The DeepEval suite in `test_rag.py` judges *answers*, needs a paid judge model,
and is skipped when no key is configured. That left the repository with no
quality gate on the majority of pushes. This suite closes that gap: it scores
*retrieval* against a fixed labelled corpus using a deterministic local
embedder, so it runs on every push, costs nothing, and cannot flake.

It exercises the real chunker, real store and real retriever — only the
embedding provider is swapped, which the Embedder protocol exists to allow.

Thresholds sit just under measured behaviour (recall@1 0.889, recall@3 1.000,
MRR 0.944 over 18 labelled queries). Because every input is fixed there is no
variance to absorb, so a drop below these means retrieval genuinely regressed. Verified to fail on a
constant similarity function, a truncating chunker, and a mis-ordering
retriever.

The same `metrics.compute()` powers the published badge, so the number on the
README is the number this gate asserted.
"""
from __future__ import annotations

import pytest
from corpus import DOCUMENTS, LABELLED
from metrics import Metrics, build_store, compute

MIN_RECALL_AT_1 = 0.80
MIN_RECALL_AT_3 = 1.00
MIN_MRR = 0.90


@pytest.fixture(scope="module")
def metrics() -> Metrics:
    return compute(build_store())


def test_index_is_populated() -> None:
    assert len(build_store()) == len(DOCUMENTS), "chunker or ingest dropped documents"


def test_every_question_is_labelled() -> None:
    urls = {d.url for d in DOCUMENTS}
    unknown = {g for g in LABELLED.values() if g not in urls}
    assert not unknown, f"labels point at documents not in the corpus: {unknown}"


def test_every_document_has_a_query() -> None:
    """A document nobody asks about is a distractor, not coverage.

    Guards the corpus against silent drift: adding a document without a query
    quietly weakens the gate instead of strengthening it.
    """
    asked = set(LABELLED.values())
    unreachable = sorted(d.url for d in DOCUMENTS if d.url not in asked)
    assert not unreachable, f"documents with no labelled query: {unreachable}"


def test_recall_at_1(metrics: Metrics) -> None:
    assert metrics.recall_at_1 >= MIN_RECALL_AT_1, (
        f"recall@1 {metrics.recall_at_1:.3f} < {MIN_RECALL_AT_1}"
    )


def test_recall_at_3(metrics: Metrics) -> None:
    assert metrics.recall_at_3 >= MIN_RECALL_AT_3, (
        f"recall@3 {metrics.recall_at_3:.3f} < {MIN_RECALL_AT_3}; "
        f"missed: {metrics.misses}"
    )


def test_mean_reciprocal_rank(metrics: Metrics) -> None:
    assert metrics.mrr >= MIN_MRR, f"MRR {metrics.mrr:.3f} < {MIN_MRR}"
