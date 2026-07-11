"""Central configuration. Provider-agnostic: works with OpenAI, GitHub Models,
Azure, or a local OpenAI-compatible server (Ollama, vLLM, LocalAI) by setting
OPENAI_BASE_URL. Nothing here is specific to a single vendor."""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "data" / "index.json"

# --- Content sources (RSS/Atom feeds to index) ---
# Comma-separated in the FEEDS env var; defaults to Santosh's Medium feed.
FEEDS = [
    f.strip()
    for f in os.getenv("FEEDS", "https://medium.com/feed/@santosh-shinde").split(",")
    if f.strip()
]

# --- Models (override per provider) ---
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# --- Retrieval ---
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))


def client():
    """Return an OpenAI-compatible client. Set OPENAI_BASE_URL to point at
    GitHub Models (https://models.inference.ai.azure.com) or a local server."""
    from openai import OpenAI

    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("GITHUB_TOKEN", ""),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
