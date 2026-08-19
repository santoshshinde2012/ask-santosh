"""OpenAI-compatible adapters for Embedder and ChatClient.

The concrete client is injected (built once in the composition root), so these
classes hold no global state and are easy to fake in tests. Because the client
is OpenAI-compatible, the same adapters serve OpenAI, Azure, Ollama, vLLM, or a
local server (Ollama/vLLM) — the only difference is the injected base_url.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class OpenAIEmbedder:
    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        return [item.embedding for item in resp.data]


class OpenAIChatClient:
    def __init__(self, client: Any, model: str, temperature: float = 0.2) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
