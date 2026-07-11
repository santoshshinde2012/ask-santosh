import pytest

from ask_santosh.models import Chunk
from ask_santosh.retriever import Retriever
from ask_santosh.service import RagService
from ask_santosh.store import JSONVectorStore

from .fakes import EchoChat, FixedEmbedder


def _store() -> JSONVectorStore:
    return JSONVectorStore(
        [
            Chunk("a", "Article A", "http://x/a", "alpha", [1.0, 0.0]),
            Chunk("b", "Article A", "http://x/a", "beta", [0.9, 0.1]),  # same url
            Chunk("c", "Article C", "http://x/c", "gamma", [0.0, 1.0]),
        ]
    )


def _service(chat: EchoChat) -> RagService:
    retriever = Retriever(FixedEmbedder([1.0, 0.0]), _store(), top_k=3)
    return RagService(retriever, chat)


def test_answer_returns_grounded_text_and_context():
    chat = EchoChat("answer text")
    ans = _service(chat).answer("what is alpha?")
    assert ans.text == "answer text"
    assert ans.retrieval_context  # non-empty
    assert "Excerpts:" in (chat.last_user or "")  # prompt was grounded


def test_sources_are_deduplicated_by_url():
    ans = _service(EchoChat()).answer("q")
    urls = [s.url for s in ans.sources]
    assert urls == ["http://x/a", "http://x/c"]  # two 'a' chunks collapse to one


def test_empty_question_raises():
    with pytest.raises(ValueError):
        _service(EchoChat()).answer("   ")
