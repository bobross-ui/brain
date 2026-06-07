import json
from pathlib import Path

import pytest

from brain.config import settings
from brain.embeddings import SentenceTransformerEmbedder
from brain.llm import LLMClient, OllamaLLMClient
from brain.memory import MemoryService
from brain.models import (
    FactCandidate,
    Memory,
    MemoryAction,
    MemoryActionKind,
    Reconciler,
    Scope,
    ScoredMemory,
)
from brain.reconcile import LLMReconciler
from brain.store.sqlite import SQLiteMemoryStore


FIXTURE_DIR = Path(__file__).parent / "fixtures"
PIZZA_TO_SUSHI_PATH = FIXTURE_DIR / "conversations" / "pizza_to_sushi.json"
RECONCILE_RESPONSE_PATH = (
    FIXTURE_DIR / "llm_responses" / "reconcile_pizza_sushi.json"
)


class StubLLM(LLMClient):
    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat_json(
        self,
        messages: list[dict],
        schema: dict,
        temperature: float = 0.0,
    ) -> dict:
        self.calls.append(
            {
                "messages": messages,
                "schema": schema,
                "temperature": temperature,
            }
        )
        if not self._responses:
            raise AssertionError("StubLLM received more calls than expected")
        response = self._responses.pop(0)
        return response.get("response", response)


class RecordingReconciler(Reconciler):
    def __init__(self, inner: Reconciler):
        self._inner = inner
        self.actions: list[MemoryAction] = []
        self.similar_memories: list[list[ScoredMemory]] = []

    async def reconcile(
        self,
        candidate: FactCandidate,
        similar_memories: list[ScoredMemory],
    ) -> MemoryAction:
        action = await self._inner.reconcile(candidate, similar_memories)
        self.actions.append(action)
        self.similar_memories.append(similar_memories)
        return action


def _load_json(path: Path):
    return json.loads(path.read_text())


def _memory(content: str, id: str = "memory-1") -> Memory:
    return Memory(
        id=id,
        content=content,
        user_id="test-user",
        metadata={},
        content_hash=f"hash-{id}",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


def _scored_memory(content: str, id: str = "memory-1") -> ScoredMemory:
    return ScoredMemory(memory=_memory(content, id), score=0.85)


async def test_update_decision():
    pizza = _scored_memory("User likes pizza")
    llm = StubLLM(
        [
            {
                "action": "UPDATE",
                "target_index": 0,
                "content": "User prefers sushi",
                "reason": "The preference changed.",
            }
        ]
    )
    reconciler = LLMReconciler(llm)

    action = await reconciler.reconcile(
        FactCandidate(content="User prefers sushi"),
        [pizza],
    )

    assert action.kind == MemoryActionKind.UPDATE
    assert action.target_id == pizza.memory.id
    assert action.content == "User prefers sushi"


async def test_delete_decision():
    pizza = _scored_memory("User likes pizza")
    llm = StubLLM(
        [
            {
                "action": "DELETE",
                "target_index": 0,
                "content": None,
                "reason": "The user invalidated this memory.",
            }
        ]
    )
    reconciler = LLMReconciler(llm)

    action = await reconciler.reconcile(
        FactCandidate(content="User no longer likes pizza"),
        [pizza],
    )

    assert action.kind == MemoryActionKind.DELETE
    assert action.target_id == pizza.memory.id
    assert action.content is None


async def test_noop_decision():
    pizza = _scored_memory("User likes pizza")
    llm = StubLLM(
        [
            {
                "action": "NOOP",
                "target_index": 0,
                "content": None,
                "reason": "The memory already captures the fact.",
            }
        ]
    )
    reconciler = LLMReconciler(llm)

    action = await reconciler.reconcile(
        FactCandidate(content="User likes pizza"),
        [pizza],
    )

    assert action.kind == MemoryActionKind.NOOP
    assert action.content is None


async def test_add_decision_empty_candidates():
    llm = StubLLM(
        [
            {
                "action": "ADD",
                "target_index": None,
                "content": "User likes tacos",
                "reason": "This is a new preference.",
            }
        ]
    )
    reconciler = LLMReconciler(llm)

    action = await reconciler.reconcile(
        FactCandidate(content="User likes tacos"),
        [],
    )

    assert action.kind == MemoryActionKind.ADD
    assert action.target_id is None
    assert action.content == "User likes tacos"


async def test_out_of_range_target_index_falls_back_to_add():
    llm = StubLLM(
        [
            {
                "action": "UPDATE",
                "target_index": 99,
                "content": "User prefers sushi",
                "reason": "The requested target is invalid.",
            }
        ]
    )
    reconciler = LLMReconciler(llm)

    action = await reconciler.reconcile(
        FactCandidate(content="User prefers sushi"),
        [_scored_memory("User likes pizza")],
    )

    assert action.kind == MemoryActionKind.ADD
    assert action.target_id is None


async def test_pizza_sushi_update_integration(store):
    scope = Scope(user_id="test-user")
    pizza = await store.add("User likes pizza", scope)
    llm = StubLLM(_load_json(RECONCILE_RESPONSE_PATH))
    reconciler = RecordingReconciler(LLMReconciler(llm))
    service = MemoryService(store, llm, reconciler, search_k=5)

    returned = await service.add(_load_json(PIZZA_TO_SUSHI_PATH), scope)

    assert len(reconciler.actions) == 1
    action = reconciler.actions[0]
    assert action.kind == MemoryActionKind.UPDATE
    assert action.target_id == pizza.id
    assert action.content == "User prefers sushi"
    assert len(reconciler.similar_memories[0]) == 1
    assert reconciler.similar_memories[0][0].memory.id == pizza.id

    assert len(returned) == 1
    assert returned[0].id == pizza.id
    assert returned[0].content == "User prefers sushi"

    fetched = await store.get(pizza.id, scope)
    assert fetched is not None
    assert fetched.id == pizza.id
    assert fetched.content == "User prefers sushi"
    assert fetched.updated_at >= pizza.updated_at

    results = await store.search("food preference", scope, limit=10)
    assert len(results) == 1
    assert results[0].memory.id == pizza.id
    assert results[0].memory.content == "User prefers sushi"
    assert all("pizza" not in result.memory.content.lower() for result in results)


@pytest.mark.live
@pytest.mark.ollama
@pytest.mark.slow
async def test_layer3_live_smoke(tmp_path):
    db_path = str(tmp_path / "live_brain.db")
    embedder = SentenceTransformerEmbedder(settings.brain_embedder_model)
    store = await SQLiteMemoryStore.create(db_path, embedder)
    llm = OllamaLLMClient(settings.llm_model)
    service = MemoryService(store, llm, LLMReconciler(llm), search_k=5)
    scope = Scope(user_id="live-user")

    await store.add("User likes pizza", scope)
    returned = await service.add(_load_json(PIZZA_TO_SUSHI_PATH), scope)

    assert len(returned) >= 1
    for memory in returned:
        assert isinstance(memory.id, str)
        assert memory.id
        assert isinstance(memory.content, str)
        assert memory.content
        assert memory.user_id == scope.user_id
