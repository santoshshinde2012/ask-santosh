"""Text chunking. Pure and deterministic — no I/O, trivially unit-testable."""
from __future__ import annotations


class SlidingWindowChunker:
    """Fixed-size character windows with overlap.

    A single responsibility: given text, return passages. Chosen for zero
    dependencies and determinism; swap in a token- or sentence-aware chunker
    later without changing any caller (it satisfies the Chunker protocol).
    """

    def __init__(self, size: int = 900, overlap: int = 150) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if not 0 <= overlap < size:
            raise ValueError("overlap must be in [0, size)")
        self._size = size
        self._step = size - overlap

    def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        return [text[i : i + self._size] for i in range(0, len(text), self._step)]
