from pathlib import Path

import pytest
import pytest_asyncio

from brain.config import settings
from brain.embeddings import FakeEmbedder
from brain.store.sqlite import SQLiteMemoryStore


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Pin ``ollama``-marked live-smoke tests to the local Ollama backend.

    They construct ``OllamaLLMClient(settings.llm_model)`` directly, so they only
    make sense when the configured provider is Ollama. Skip them otherwise (e.g.
    when ``LLM_PROVIDER=deepseek``) instead of failing with a model-not-found error.
    """
    if "ollama" in item.keywords and settings.llm_provider != "ollama":
        pytest.skip(
            f"ollama-only live-smoke test (LLM_PROVIDER={settings.llm_provider!r}, "
            "not 'ollama')"
        )


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest_asyncio.fixture
async def store(tmp_path: Path, fake_embedder: FakeEmbedder) -> SQLiteMemoryStore:
    db_path = str(tmp_path / "test_brain.db")
    return await SQLiteMemoryStore.create(db_path, fake_embedder)
