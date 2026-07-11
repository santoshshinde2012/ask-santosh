import pytest

from ask_santosh.chunking import SlidingWindowChunker


def test_splits_with_overlap():
    chunker = SlidingWindowChunker(size=10, overlap=4)
    chunks = chunker.split("abcdefghijklmnop")  # 16 chars, step = 6
    assert chunks == ["abcdefghij", "ghijklmnop", "mnop"]


def test_short_text_is_single_chunk():
    assert SlidingWindowChunker(size=100, overlap=10).split("hello world") == ["hello world"]


def test_empty_text_returns_no_chunks():
    assert SlidingWindowChunker().split("   ") == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (10, 10), (10, 11)])
def test_invalid_config_raises(size, overlap):
    with pytest.raises(ValueError):
        SlidingWindowChunker(size=size, overlap=overlap)
