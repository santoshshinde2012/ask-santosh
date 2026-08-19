"""A deterministic, dependency-free Embedder for the key-free quality gate.

The point of the protocol-based design is that `Embedder` is an abstraction, so
retrieval can be exercised end to end without OpenAI, a network, or a spend.
This is a hashed bag-of-words projection: tokens are hashed into a fixed number
of buckets and weighted by sublinear term frequency, then L2-normalised so the
store's cosine scan behaves exactly as it does with real embeddings.

It is deliberately *not* good enough to replace a real embedding model. It is
good enough to catch the regressions that matter in CI: a chunker that drops
text, a store that scores wrongly, a retriever that returns the wrong passage.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")

# Words carrying no retrieval signal. Without IDF the hashed space is dominated
# by these, which would flatter the metrics and make the gate meaningless.
_STOP = frozenset("""
a an and are as at be but by for from has have how in into is it its of on or
that the their then there these this to was were what when where which who why
with you your i we our us they them he she his her do does did done can could
should would will just about over under more most some any each other
""".split())

DIMENSIONS = 512


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _bucket(token: str, dimensions: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


class HashingEmbedder:
    """Implements the Embedder protocol with no third-party dependency."""

    def __init__(self, dimensions: int = DIMENSIONS) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        counts = Counter(_tokens(text))
        for token, count in counts.items():
            # Sublinear TF: a term repeated 10x is not 10x more important.
            vector[_bucket(token, self._dimensions)] += 1.0 + math.log(count)
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]
