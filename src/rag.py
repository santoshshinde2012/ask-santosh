"""Retrieval-augmented answering over the ingested writing.

    from src.rag import answer
    result = answer("What does Santosh think about RAG evals?")
    print(result["answer"], result["sources"])
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

from . import config

SYSTEM = (
    "You are Santosh Shinde's writing assistant. Answer questions strictly from "
    "the provided excerpts of his published articles. Be concise and technical. "
    "If the excerpts do not cover the question, say so plainly rather than "
    "guessing. Never invent facts, numbers, or sources."
)


@lru_cache(maxsize=1)
def _index() -> list[dict]:
    if not config.INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{config.INDEX_PATH} missing — run `python -m src.ingest` first."
        )
    return json.loads(config.INDEX_PATH.read_text())


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(question: str, k: int | None = None) -> list[dict]:
    k = k or config.TOP_K
    qvec = (
        config.client()
        .embeddings.create(model=config.EMBED_MODEL, input=[question])
        .data[0]
        .embedding
    )
    ranked = sorted(
        _index(), key=lambda r: _cosine(qvec, r["embedding"]), reverse=True
    )
    return ranked[:k]


def answer(question: str, k: int | None = None) -> dict:
    hits = retrieve(question, k)
    context = "\n\n".join(
        f"[{i + 1}] {h['title']}\n{h['text']}" for i, h in enumerate(hits)
    )
    resp = config.client().chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"},
        ],
    )
    seen, sources = set(), []
    for h in hits:
        if h["url"] and h["url"] not in seen:
            seen.add(h["url"])
            sources.append({"title": h["title"], "url": h["url"]})
    return {
        "answer": resp.choices[0].message.content.strip(),
        "sources": sources,
        "retrieval_context": [h["text"] for h in hits],
    }
