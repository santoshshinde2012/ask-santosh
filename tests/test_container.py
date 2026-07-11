import pytest

from ask_santosh.container import build_in_memory_rag_service, build_openai_client
from ask_santosh.settings import Settings


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)


def test_client_build_fails_fast_without_key():
    with pytest.raises(RuntimeError, match="No API key"):
        build_openai_client(Settings.from_env())


def test_in_memory_service_requires_key():
    with pytest.raises(RuntimeError, match="No API key"):
        build_in_memory_rag_service(Settings.from_env())
