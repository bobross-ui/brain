import json
from pathlib import Path

import pytest

from brain.config import settings
from brain.extract import Extractor
from brain.llm import FakeLLMClient, OllamaLLMClient
from brain.memory import MemoryService
from brain.models import Scope, Turn
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


def _turns(messages: list[dict]) -> list[Turn]:
    return [
        Turn(
            speaker=message["role"],
            text=message["content"],
            source_turn_id=f"message-{index}",
        )
        for index, message in enumerate(messages)
    ]


async def test_extract_oracle():
    messages = _load_json(CONVERSATION_PATH)
    recorded = _load_json(RECORDED_RESPONSE_PATH)
    extractor = Extractor(FakeLLMClient(recorded=recorded))

    candidates = await extractor.extract(
        _turns(messages),
        roster={"primary": "user", "assistant": "assistant"},
        session_observed_at="2026-06-10T00:00:00+00:00",
    )

    assert len(candidates) == 5
    contents = {candidate.content for candidate in candidates}
    assert contents == EXPECTED_FACTS
    for candidate in candidates:
        assert candidate.subject == "user"
        assert candidate.source_turn_ids
        assert candidate.metadata == {
            "source": "extraction",
            "source_turn_ids": candidate.source_turn_ids,
        }


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
        assert memory.subject == "user"
        assert memory.source_turn_ids
        assert memory.source_session_id is not None


@pytest.mark.ollama
async def test_extract_live_smoke():
    messages = _load_json(CONVERSATION_PATH)
    extractor = Extractor(OllamaLLMClient(settings.llm_model))

    candidates = await extractor.extract(_turns(messages))

    assert len(candidates) >= 1
    assert isinstance(candidates[0].content, str)
    assert candidates[0].content
    assert candidates[0].metadata["source"] == "extraction"


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
