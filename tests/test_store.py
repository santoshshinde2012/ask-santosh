import pytest

from ask_santosh.models import Chunk
from ask_santosh.store import JSONVectorStore


def _chunk(cid: str, vec: list[float], url: str = "") -> Chunk:
    return Chunk(id=cid, title=cid, url=url or f"http://x/{cid}", text=cid, embedding=vec)


def test_upsert_dedups_by_id():
    store = JSONVectorStore()
    store.upsert([_chunk("a", [1, 0])])
    store.upsert([_chunk("a", [0, 1])])  # same id -> replace
    assert len(store) == 1


def test_search_ranks_by_cosine():
    store = JSONVectorStore([_chunk("a", [1, 0]), _chunk("b", [0, 1]), _chunk("c", [0.9, 0.1])])
    hits = store.search([1.0, 0.0], k=2)
    assert [h.id for h in hits] == ["a", "c"]


def test_search_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        JSONVectorStore().search([1.0], k=0)


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "index.json"
    JSONVectorStore([_chunk("a", [1, 0])]).save(path)
    restored = JSONVectorStore.load(path)
    assert len(restored) == 1
    assert restored.search([1.0, 0.0], k=1)[0].id == "a"


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        JSONVectorStore.load(tmp_path / "nope.json")
