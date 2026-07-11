"""Typed application settings, loaded from the environment in one place."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_FEED = "https://medium.com/feed/@santosh-shinde"
_ROOT = Path(__file__).resolve().parent.parent.parent


def _feeds_from_env() -> list[str]:
    raw = os.getenv("FEEDS", _DEFAULT_FEED)
    return [f.strip() for f in raw.split(",") if f.strip()]


def _api_key() -> str:
    return os.getenv("OPENAI_API_KEY") or os.getenv("GITHUB_TOKEN") or ""


@dataclass(frozen=True, slots=True)
class Settings:
    feeds: list[str] = field(default_factory=_feeds_from_env)
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    chunk_size: int = int(os.getenv("CHUNK_CHARS", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    api_key: str = field(default_factory=_api_key)
    base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    index_path: Path = _ROOT / "data" / "index.json"

    @classmethod
    def from_env(cls) -> Settings:
        return cls()
