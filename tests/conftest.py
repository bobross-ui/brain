from pathlib import Path

import pytest
import pytest_asyncio

from brain.embeddings import FakeEmbedder
from brain.store.sqlite import SQLiteMemoryStore


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest_asyncio.fixture
async def store(tmp_path: Path, fake_embedder: FakeEmbedder) -> SQLiteMemoryStore:
    db_path = str(tmp_path / "test_brain.db")
    return await SQLiteMemoryStore.create(db_path, fake_embedder)
