# ask-santosh — RAG over my writing, with an eval harness

> A PDF lists skills. A working system proves them.

A small, honest, **retrieval-augmented question-answering service** over
[Santosh Shinde's](https://github.com/santoshshinde2012) published articles, plus a
**reproducible DeepEval suite** that scores answer quality on every run. It exists to
*demonstrate* — not just claim — hands-on RAG, evaluation, and MLOps practice.

It is deliberately dependency-light and **provider-agnostic**: point it at OpenAI,
GitHub Models, or a local server (Ollama / vLLM / LocalAI) by changing two env vars.

---

## How it works

```
Medium RSS ──► ingest ──► chunk ──► embed ──► data/index.json (vector store)
                                                   │
question ──► embed ──► cosine top-k ──► prompt ──► chat model ──► answer + cited sources
                                                   │
                                            DeepEval: faithfulness · answer-relevancy · contextual-relevancy
```

- **`src/ingest.py`** — pulls the configured feeds, strips HTML, chunks (~900 chars,
  150 overlap), embeds each chunk, and writes a flat JSON index.
- **`src/rag.py`** — embeds the query, ranks chunks by cosine similarity, builds a
  grounded prompt, and returns `{answer, sources, retrieval_context}`. The system
  prompt forbids answering outside the retrieved excerpts (no hallucinated facts).
- **`src/app.py`** — a FastAPI service exposing `GET/POST /ask`.
- **`evals/test_rag.py`** — runs the real pipeline over a question set and asserts
  three reference-free RAG metrics via an LLM judge, at a 0.7 threshold, in CI.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env            # add OPENAI_API_KEY (or point at GitHub Models / Ollama)

python -m src.ingest            # build the vector index
uvicorn src.app:app --reload    # serve
curl -s "localhost:8000/ask?q=What+is+FrameSleuth%3F"
```

## Evaluation

```bash
deepeval test run evals/test_rag.py
```

Every question is scored on **Answer Relevancy**, **Faithfulness** (grounded in the
retrieved context — the anti-hallucination check), and **Contextual Relevancy**
(did retrieval surface the right passages). The same suite runs in
[`.github/workflows/evals.yml`](.github/workflows/evals.yml) on every push, gating at
0.7 — so answer quality is a measured, versioned property, not a vibe.

## Design choices

- **Grounded-or-abstain** system prompt over cheap fluency — the model is instructed
  to say "not covered" rather than guess.
- **Reference-free metrics** so the suite keeps working as new articles are ingested,
  without hand-labelling golden answers.
- **Portable** — swap providers with `OPENAI_BASE_URL`; run fully local with Ollama.

## License

Apache-2.0.
