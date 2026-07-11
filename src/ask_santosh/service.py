"""The RAG use-case: retrieve grounded context, then answer with citations.

Orchestrates a Retriever and a ChatClient through their protocols only. This is
the one place business rules live (grounded-or-abstain prompting, source
dedup) — deliberately free of any provider or transport detail.
"""
from __future__ import annotations

from .models import Answer, Chunk, Source
from .protocols import ChatClient
from .retriever import Retriever

SYSTEM_PROMPT = (
    "You are Santosh Shinde's writing assistant. Answer strictly from the "
    "provided excerpts of his published articles. Be concise and technical. If "
    "the excerpts do not cover the question, say so plainly rather than "
    "guessing. Never invent facts, numbers, or sources."
)


class RagService:
    def __init__(self, retriever: Retriever, chat: ChatClient) -> None:
        self._retriever = retriever
        self._chat = chat

    def answer(self, question: str, k: int | None = None) -> Answer:
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        hits = self._retriever.retrieve(question, k)
        user = f"Excerpts:\n{self._format(hits)}\n\nQuestion: {question.strip()}"
        text = self._chat.complete(SYSTEM_PROMPT, user)
        return Answer(
            text=text.strip(),
            sources=self._dedup_sources(hits),
            retrieval_context=tuple(h.text for h in hits),
        )

    @staticmethod
    def _format(hits: list[Chunk]) -> str:
        return "\n\n".join(f"[{i + 1}] {h.title}\n{h.text}" for i, h in enumerate(hits))

    @staticmethod
    def _dedup_sources(hits: list[Chunk]) -> tuple[Source, ...]:
        seen: set[str] = set()
        out: list[Source] = []
        for h in hits:
            if h.url and h.url not in seen:
                seen.add(h.url)
                out.append(Source(title=h.title, url=h.url))
        return tuple(out)
