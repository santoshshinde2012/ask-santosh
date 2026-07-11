---
title: Ask Santosh
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: RAG over my writing, with cited sources.
---

# Ask Santosh

A live, retrieval-augmented chatbot over
[Santosh Shinde's](https://github.com/santoshshinde2012) published writing. Every
answer is grounded in retrieved excerpts and cites its sources.

- **Code & architecture:** [github.com/santoshshinde2012/ask-santosh](https://github.com/santoshshinde2012/ask-santosh)
- **Config:** set `OPENAI_API_KEY` in **Settings › Secrets** (or point at GitHub
  Models / a local model via `OPENAI_BASE_URL`, `CHAT_MODEL`, `EMBED_MODEL`).

The index is built in memory on first request from the configured `FEEDS`
(defaults to the Medium feed), so no data is bundled with the Space.
