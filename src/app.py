"""Minimal FastAPI service exposing the RAG pipeline.

    uvicorn src.app:app --reload
    curl -s "localhost:8000/ask?q=What+is+FrameSleuth%3F"
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .rag import answer

app = FastAPI(title="Ask Santosh", version="0.1.0")


class Query(BaseModel):
    q: str


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/ask")
def ask_get(q: str) -> dict:
    return answer(q)


@app.post("/ask")
def ask_post(query: Query) -> dict:
    return answer(query.q)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (
        "<h2>Ask Santosh</h2>"
        "<p>RAG over Santosh Shinde's writing. "
        "Try <code>/ask?q=your+question</code>.</p>"
    )
