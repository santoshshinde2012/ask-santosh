# ask-santosh — RAG over my writing, with an eval harness

> A PDF lists skills. A working, tested system proves them.

[![CI](https://github.com/santoshshinde2012/ask-santosh/actions/workflows/ci.yml/badge.svg)](https://github.com/santoshshinde2012/ask-santosh/actions/workflows/ci.yml)

A small, honest **retrieval-augmented question-answering** service over
[Santosh Shinde's](https://github.com/santoshshinde2012) published articles, plus a
**reproducible DeepEval suite** that scores answer quality in CI. It exists to
*demonstrate* — not just claim — hands-on RAG, evaluation, and clean engineering.

Dependency-light and **provider-agnostic**: point it at OpenAI, GitHub Models, or a
local server (Ollama / vLLM / LocalAI) by changing two env vars.

---

## Architecture (SOLID by construction)

Business logic depends on **abstractions**, never on a vendor. Concrete providers are
wired in exactly one place — the composition root.

```
DocumentSource ─┐
     Chunker ───┤ IngestionPipeline ──► VectorStore.upsert ──► data/index.json
    Embedder ───┘

   question ──► Retriever(Embedder, VectorStore) ──► RagService(Retriever, ChatClient) ──► Answer + sources
                                                          │
                                              DeepEval: faithfulness · answer-relevancy · contextual-relevancy
```

| Principle | Where |
| --- | --- |
| **S**ingle responsibility | one job per module — `chunking`, `store`, `retriever`, `service`, `feeds`, `openai_adapter` |
| **O**pen/closed | new provider or vector store = new class implementing a protocol; callers unchanged |
| **L**iskov | any `VectorStore` (JSON today, pgvector tomorrow) drops in without breaking `Retriever` |
| **I**nterface segregation | narrow protocols: `Embedder`, `ChatClient`, `Chunker`, `DocumentSource`, `VectorStore` |
| **D**ependency inversion | `RagService`/`Retriever` depend on `protocols.py`, not on OpenAI; wiring lives in `container.py` |

The payoff: the whole pipeline is **unit-tested with fakes — no network, no API key**
(see `tests/`), while the LLM-judge evals run behind a secret.

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env               # add OPENAI_API_KEY (or point at GitHub Models / Ollama)

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

```bash
pip install -e ".[evals]"
ask-santosh-ingest
deepeval test run evals/test_rag.py
```

Each question runs the real pipeline and is scored on **Answer Relevancy**,
**Faithfulness** (grounded in retrieved context — the anti-hallucination check), and
**Contextual Relevancy**, gating at 0.7. The same suite runs in
[CI](.github/workflows/ci.yml) once `OPENAI_API_KEY` is set as a repo secret; until
then it skips cleanly so the badge stays green.

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

## License

Apache-2.0.
