"""Reproducible RAG evaluation with DeepEval (pytest-style).

Each question runs the real pipeline and is scored on three reference-free
RAG metrics via an LLM judge:

  - Answer Relevancy   — does the answer address the question?
  - Faithfulness       — is the answer grounded in retrieved excerpts (no hallucination)?
  - Contextual Relevancy — did retrieval surface relevant context?

Run:  deepeval test run evals/test_rag.py      (or: pytest evals/test_rag.py)
Requires an index (python -m src.ingest) and a judge model key.
"""
from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from src.rag import answer

THRESHOLD = 0.7

QUESTIONS = [
    "What is FrameSleuth and what problem does it solve?",
    "How does Santosh approach evaluating RAG systems in production?",
    "What tradeoffs matter when designing a production MCP server?",
    "What is the medallion architecture and where does it fall short?",
    "Why does Santosh say the model is the easy part?",
]


@pytest.mark.parametrize("question", QUESTIONS)
def test_rag_quality(question: str) -> None:
    result = answer(question)
    case = LLMTestCase(
        input=question,
        actual_output=result["answer"],
        retrieval_context=result["retrieval_context"],
    )
    assert_test(
        case,
        [
            AnswerRelevancyMetric(threshold=THRESHOLD),
            FaithfulnessMetric(threshold=THRESHOLD),
            ContextualRelevancyMetric(threshold=THRESHOLD),
        ],
    )
