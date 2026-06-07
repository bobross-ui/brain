import json
from pathlib import Path

import pytest

from brain.config import settings
from brain.extract import Extractor
from brain.llm import FakeLLMClient, OllamaLLMClient
from brain.memory import MemoryService
from brain.models import Scope
from brain.reconcile import AlwaysAddReconciler


FIXTURE_DIR = Path(__file__).parent / "fixtures"
CONVERSATION_PATH = FIXTURE_DIR / "conversations" / "conv_01.json"
RECORDED_RESPONSE_PATH = FIXTURE_DIR / "llm_responses" / "extract_conv_01.json"
EXPECTED_FACTS = {
    "The user moved to Berlin last month.",
    "The user works as a software engineer.",
    "The user cycles to work every day.",
    "The user plays chess in the evenings.",
    "The user is vegetarian.",
}


def _load_json(path: Path):
    return json.loads(path.read_text())


async def test_extract_oracle():
    messages = _load_json(CONVERSATION_PATH)
    recorded = _load_json(RECORDED_RESPONSE_PATH)
    extractor = Extractor(FakeLLMClient(recorded=recorded))

    candidates = await extractor.extract(messages)

    assert len(candidates) == 5
    contents = {candidate.content for candidate in candidates}
    assert contents == EXPECTED_FACTS
    for candidate in candidates:
        assert candidate.metadata == {"source": "extraction"}


async def test_memory_service_add_oracle(store):
    messages = _load_json(CONVERSATION_PATH)
    recorded = _load_json(RECORDED_RESPONSE_PATH)
    service = MemoryService(
        store,
        FakeLLMClient(recorded=recorded),
        AlwaysAddReconciler(),
    )
    scope = Scope(user_id="test-user")

    stored = await service.add(messages, scope)

    assert len(stored) == 5
    stored_contents = {memory.content for memory in stored}
    assert stored_contents == EXPECTED_FACTS
    for memory in stored:
        assert isinstance(memory.id, str)
        assert memory.user_id == "test-user"
        assert memory.namespace == "default"


@pytest.mark.ollama
async def test_extract_live_smoke():
    messages = _load_json(CONVERSATION_PATH)
    extractor = Extractor(OllamaLLMClient(settings.llm_model))

    candidates = await extractor.extract(messages)

    assert len(candidates) >= 1
    assert isinstance(candidates[0].content, str)
    assert candidates[0].content
    assert candidates[0].metadata == {"source": "extraction"}


@pytest.mark.ollama
async def test_memory_service_add_live_smoke(store):
    messages = _load_json(CONVERSATION_PATH)
    llm = OllamaLLMClient(settings.llm_model)
    service = MemoryService(store, llm, AlwaysAddReconciler())
    scope = Scope(user_id="live-user")

    stored = await service.add(messages, scope)

    assert len(stored) >= 1
    assert isinstance(stored[0].content, str)
    assert stored[0].content
