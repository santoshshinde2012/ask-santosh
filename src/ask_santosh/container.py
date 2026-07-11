"""Composition root: the single place concrete implementations are wired.

Every other module depends on protocols; only here do we name OpenAI, the RSS
source, and the JSON store. Swapping a provider or store is a change confined to
this file.
"""
from __future__ import annotations

from typing import Any

from .chunking import SlidingWindowChunker
from .feeds import RssDocumentSource
from .ingest import IngestionPipeline
from .openai_adapter import OpenAIChatClient, OpenAIEmbedder
from .retriever import Retriever
from .service import RagService
from .settings import Settings
from .store import JSONVectorStore


def build_openai_client(settings: Settings) -> Any:
    from openai import OpenAI  # lazy import

    if not settings.api_key:
        raise RuntimeError(
            "No API key found. Set OPENAI_API_KEY (or GITHUB_TOKEN with "
            "OPENAI_BASE_URL for GitHub Models)."
        )
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)


def build_ingestion_pipeline(settings: Settings) -> IngestionPipeline:
    client = build_openai_client(settings)
    return IngestionPipeline(
        source=RssDocumentSource(settings.feeds),
        chunker=SlidingWindowChunker(settings.chunk_size, settings.chunk_overlap),
        embedder=OpenAIEmbedder(client, settings.embed_model),
    )


def build_store_for_write() -> JSONVectorStore:
    return JSONVectorStore()


def build_rag_service(settings: Settings) -> RagService:
    client = build_openai_client(settings)
    store = JSONVectorStore.load(settings.index_path)
    retriever = Retriever(
        embedder=OpenAIEmbedder(client, settings.embed_model),
        store=store,
        top_k=settings.top_k,
    )
    chat = OpenAIChatClient(client, settings.chat_model, settings.temperature)
    return RagService(retriever, chat)
