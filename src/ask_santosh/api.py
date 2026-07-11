"""FastAPI transport. Thin adapter over RagService — no business logic here.

The service is built once at startup (lazily, so importing this module for tests
or docs never requires an API key or a prebuilt index).
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .container import build_rag_service
from .service import RagService
from .settings import Settings

app = FastAPI(title="Ask Santosh", version="1.0.0")


class Query(BaseModel):
    q: str


@lru_cache(maxsize=1)
def _service() -> RagService:
    return build_rag_service(Settings.from_env())


def _serialize(question: str) -> dict:
    ans = _service().answer(question)
    return {
        "answer": ans.text,
        "sources": [{"title": s.title, "url": s.url} for s in ans.sources],
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/ask")
def ask_get(q: str) -> dict:
    return _serialize(q)


@app.post("/ask")
def ask_post(query: Query) -> dict:
    return _serialize(query.q)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (
        "<h2>Ask Santosh</h2><p>RAG over Santosh Shinde's writing. "
        "Try <code>/ask?q=your+question</code>.</p>"
    )
