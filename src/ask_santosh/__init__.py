"""ask-santosh: RAG over my writing, wired through small SOLID components."""
from __future__ import annotations

from .models import Answer, Chunk, Document, Source
from .retriever import Retriever
from .service import RagService

__all__ = ["Answer", "Chunk", "Document", "Source", "Retriever", "RagService"]
__version__ = "1.0.0"
