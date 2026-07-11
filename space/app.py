"""Hugging Face Space — a live chat UI over the ask-santosh RAG service.

The Space installs the package straight from GitHub (see requirements.txt), so
this file stays a thin transport: build the service once, answer questions,
cite sources. Set OPENAI_API_KEY in the Space's secrets.
"""
from __future__ import annotations

import gradio as gr

from ask_santosh.container import build_in_memory_rag_service
from ask_santosh.service import RagService
from ask_santosh.settings import Settings

MAX_QUESTION_CHARS = 500  # simple guardrail against abuse / runaway cost

_service: RagService | None = None
_error: str | None = None


def _get_service() -> RagService | None:
    """Lazily build (and cache) the service on first use."""
    global _service, _error
    if _service is None and _error is None:
        try:
            _service = build_in_memory_rag_service(Settings.from_env())
        except Exception as exc:  # surface config errors in the UI, don't crash
            _error = str(exc)
    return _service


def respond(message: str, history: list) -> str:
    svc = _get_service()
    if svc is None:
        return (
            "⚠️ The demo isn't configured yet. Set `OPENAI_API_KEY` in this Space's "
            f"**Settings › Secrets**.\n\n_Detail: {_error}_"
        )
    question = (message or "").strip()[:MAX_QUESTION_CHARS]
    if not question:
        return "Ask me anything about Santosh's writing — MCP, RAG, evals, Databricks…"
    answer = svc.answer(question)
    if not answer.sources:
        return answer.text
    sources = "\n".join(f"- [{s.title}]({s.url})" for s in answer.sources)
    return f"{answer.text}\n\n**Sources**\n{sources}"


demo = gr.ChatInterface(
    fn=respond,
    title="Ask Santosh",
    description=(
        "Retrieval-augmented Q&A over Santosh Shinde's published writing. "
        "Answers are grounded in retrieved excerpts and cite their sources. "
        "Code on [GitHub](https://github.com/santoshshinde2012/ask-santosh)."
    ),
    examples=[
        "What is FrameSleuth and what problem does it solve?",
        "How does Santosh think about evaluating RAG systems?",
        "What tradeoffs matter when designing a production MCP server?",
        "Why does he say the model is the easy part?",
    ],
)
demo.queue(default_concurrency_limit=2, max_size=32)

if __name__ == "__main__":
    demo.launch()
