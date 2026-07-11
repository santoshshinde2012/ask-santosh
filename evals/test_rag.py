"""Reproducible RAG evaluation with DeepEval (LLM-as-judge).

Gated: skipped unless a judge key is configured, so the default CI stays green.
When enabled it runs the *real* pipeline and scores three reference-free RAG
metrics — Answer Relevancy, Faithfulness (anti-hallucination), and Contextual
Relevancy — at a 0.7 threshold.

    deepeval test run evals/test_rag.py
"""
from __future__ import annotations

import os

import pytest

_HAS_KEY = bool(os.getenv("OPENAI_API_KEY") or os.getenv("GITHUB_TOKEN"))
pytestmark = pytest.mark.skipif(not _HAS_KEY, reason="no judge key configured")

THRESHOLD = 0.7
QUESTIONS = [
    "What is FrameSleuth and what problem does it solve?",
    "How does Santosh approach evaluating RAG systems in production?",
    "What tradeoffs matter when designing a production MCP server?",
    "What is the medallion architecture and where does it fall short?",
    "Why does Santosh say the model is the easy part?",
]


@pytest.fixture(scope="module")
def service():
    from ask_santosh.container import build_rag_service
    from ask_santosh.settings import Settings

    return build_rag_service(Settings.from_env())


@pytest.mark.parametrize("question", QUESTIONS)
def test_rag_quality(question: str, service) -> None:
    from deepeval import assert_test
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    ans = service.answer(question)
    case = LLMTestCase(
        input=question,
        actual_output=ans.text,
        retrieval_context=list(ans.retrieval_context),
    )
    assert_test(
        case,
        [
            AnswerRelevancyMetric(threshold=THRESHOLD),
            FaithfulnessMetric(threshold=THRESHOLD),
            ContextualRelevancyMetric(threshold=THRESHOLD),
        ],
    )
