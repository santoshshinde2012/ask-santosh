"""Deterministic in-memory fakes — the payoff of protocol-based design.

Because collaborators are abstractions, the whole pipeline is testable with no
network, no API key, and no third-party libraries.
"""
from __future__ import annotations

from collections.abc import Sequence


class FixedEmbedder:
    """Returns a fixed query vector; ignores input text."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


class EchoChat:
    """Records the last prompt and returns a canned answer."""

    def __init__(self, reply: str = "grounded answer") -> None:
        self.reply = reply
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(self, system: str, user: str) -> str:
        self.last_system, self.last_user = system, user
        return self.reply
