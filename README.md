# ask-santosh — RAG over my writing, with an eval harness

> A PDF lists skills. A working, tested system proves them.

[![CI](https://github.com/santoshshinde2012/ask-santosh/actions/workflows/ci.yml/badge.svg)](https://github.com/santoshshinde2012/ask-santosh/actions/workflows/ci.yml)
[![recall@1](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/santoshshinde2012/ask-santosh/gh-pages/metrics/recall1.json)](https://raw.githubusercontent.com/santoshshinde2012/ask-santosh/gh-pages/metrics/summary.json)
[![recall@3](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/santoshshinde2012/ask-santosh/gh-pages/metrics/recall3.json)](https://raw.githubusercontent.com/santoshshinde2012/ask-santosh/gh-pages/metrics/summary.json)
[![MRR](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/santoshshinde2012/ask-santosh/gh-pages/metrics/mrr.json)](https://raw.githubusercontent.com/santoshshinde2012/ask-santosh/gh-pages/metrics/summary.json)
[![evals updated](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/santoshshinde2012/ask-santosh/gh-pages/metrics/updated.json)](https://raw.githubusercontent.com/santoshshinde2012/ask-santosh/gh-pages/metrics/summary.json)

<sub>Those four are not decoration: CI recomputes them on every push to `main` and
publishes the result, so they always show what the quality gate actually measured.
The raw numbers are in [summary.json](https://raw.githubusercontent.com/santoshshinde2012/ask-santosh/gh-pages/metrics/summary.json).</sub>

A small, honest **retrieval-augmented question-answering** service over
[Santosh Shinde's](https://github.com/santoshshinde2012) published articles, plus a
**reproducible DeepEval suite** that scores answer quality in CI. It exists to
*demonstrate* — not just claim — hands-on RAG, evaluation, and clean engineering.

Dependency-light and **provider-agnostic**: point it at OpenAI, Azure, or a local
server (Ollama / vLLM / LocalAI) by changing two env vars.

---

## Architecture (SOLID by construction)

Business logic depends on **abstractions**, never on a vendor. Concrete providers are
wired in exactly one place — the composition root.

```
DocumentSource ─┐
     Chunker ───┤ IngestionPipeline ──► VectorStore.upsert ──► data/index.json
    Embedder ───┘

   question ──► Retriever(Embedder, VectorStore) ──► RagService(Retriever, ChatClient) ──► Answer + sources
                        │                                 │
        recall@k / MRR ─┘                                 └─ DeepEval: faithfulness ·
        (key-free, every push)                               answer-relevancy · contextual-relevancy
```

| Principle | Where |
| --- | --- |
| **S**ingle responsibility | one job per module — `chunking`, `store`, `retriever`, `service`, `feeds`, `openai_adapter` |
| **O**pen/closed | new provider or vector store = new class implementing a protocol; callers unchanged |
| **L**iskov | any `VectorStore` (JSON today, pgvector tomorrow) drops in without breaking `Retriever` |
| **I**nterface segregation | narrow protocols: `Embedder`, `ChatClient`, `Chunker`, `DocumentSource`, `VectorStore` |
| **D**ependency inversion | `RagService`/`Retriever` depend on `protocols.py`, not on OpenAI; wiring lives in `container.py` |

The payoff is testability. Because `Embedder` is an abstraction, retrieval can be
scored end to end with a local hashing embedder — **no network, no key, no spend** —
so the quality gate runs on every push rather than only when a judge key exists.

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env               # add OPENAI_API_KEY (or point OPENAI_BASE_URL at Ollama)

ask-santosh-ingest                 # build the vector index
uvicorn ask_santosh.api:app --reload
curl -s "localhost:8000/ask?q=What+is+FrameSleuth%3F"
```

## Develop & test

```bash
ruff check .          # lint + import order
mypy src              # types
pytest                # unit tests (no API key needed)
```

## Evaluate

Two tiers, deliberately. The cheap one always runs; the expensive one runs when it can.

### 1. Retrieval gate — key-free, runs on every push

```bash
pytest evals/test_retrieval.py
```

Scores **recall@1**, **recall@3** and **MRR** over a fixed labelled corpus
(`evals/corpus.py`) using a deterministic hashing embedder, exercising the real
chunker, store and retriever. Current thresholds: recall@1 ≥ 0.80, recall@3 = 1.00,
MRR ≥ 0.90, against measured 0.857 / 1.000 / 0.929.

Nothing here is random or networked, so a failure means retrieval actually
regressed. It is verified to catch a broken similarity function, a truncating
chunker, and a mis-ordering retriever.

### 2. Answer gate — DeepEval, needs a judge key

```bash
pip install -e ".[evals]"
ask-santosh-ingest
deepeval test run evals/test_rag.py
```

Each question runs the real pipeline and is scored on **Answer Relevancy**,
**Faithfulness** (the anti-hallucination check) and **Contextual Relevancy**, gating
at 0.7. In [CI](.github/workflows/ci.yml) this job is gated at the job level, so
without `OPENAI_API_KEY` it reports as **skipped** — a grey check, not a green one
that ran nothing.

## Deploy a live demo (Hugging Face Space)

A self-contained Gradio Space lives in [`space/`](space/) — a chat UI that installs
this package from GitHub and indexes in memory on first request.

```bash
pip install huggingface_hub
export HF_TOKEN=hf_xxx          # write token
export OPENAI_API_KEY=sk_xxx    # also stored as the Space secret
python space/deploy.py <your-hf-username>
```

Full instructions (and a web-UI path) in [`space/DEPLOY.md`](space/DEPLOY.md).

## Design choices

- **Grounded-or-abstain** prompting — the model is told to say "not covered" rather
  than guess.
- **Reference-free metrics** so the suite keeps working as new articles are ingested,
  without hand-labelling golden answers.
- **Portable** — swap providers with `OPENAI_BASE_URL`; run fully local with Ollama.

### Known limits

- **The corpus is the last 10 posts.** Medium's RSS feed returns only the ten most
  recent items, so "everything I've written" is really a rolling window. Point
  `FEEDS` at more feeds, or add a source that reads an archive, to widen it.
- **Linear cosine scan.** Correct and dependency-free at this size; swap
  `JSONVectorStore` for pgvector/FAISS when the corpus outgrows memory.

## License

Apache-2.0.
