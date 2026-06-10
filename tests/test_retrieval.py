from pathlib import Path
from typing import Sequence

import pytest

from brain.embeddings import Embedder
from brain.llm import FakeLLMClient
from brain.memory import MemoryService
from brain.models import (
    FactCandidate,
    MemoryAction,
    MemoryActionKind,
    Reconciler,
    Scope,
    ScoredMemory,
    SessionInput,
    Turn,
)
from brain.reconcile import AlwaysAddReconciler
from brain.store.sqlite import SQLiteMemoryStore


class ControlledEmbedder(Embedder):
    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    @property
    def dim(self) -> int:
        return 2

    async def embed(self, text: str) -> list[float]:
        return self._vectors.get(text, [1.0, 0.0])


class RecordingReconciler(Reconciler):
    def __init__(self):
        self.similar_memories: list[ScoredMemory] = []

    async def reconcile(
        self,
        candidate: FactCandidate,
        similar_memories: list[ScoredMemory],
    ) -> MemoryAction:
        self.similar_memories = similar_memories
        return MemoryAction(kind=MemoryActionKind.NOOP)


class StubReranker:
    async def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        return [10.0 if "preferred" in document else 0.0 for document in documents]


async def _controlled_store(
    tmp_path: Path,
    vectors: dict[str, list[float]],
) -> SQLiteMemoryStore:
    return await SQLiteMemoryStore.create(
        str(tmp_path / "retrieval.db"),
        ControlledEmbedder(vectors),
    )


async def test_hybrid_adds_exact_bm25_candidate_missed_by_vector(
    tmp_path: Path,
):
    query = "Zephyr 1997"
    exact = "Archive code Zephyr 1997 identifies the Berlin record."
    distractor = "Dense favorite unrelated memory."
    store = await _controlled_store(
        tmp_path,
        {
            query: [0.0, 1.0],
            exact: [1.0, 0.0],
            distractor: [0.0, 1.0],
        },
    )
    scope = Scope(user_id="alice")
    exact_memory = await store.add(exact, scope)
    distractor_memory = await store.add(distractor, scope)

    vector = await store.search(query, scope, limit=1, mode="vector")
    hybrid = await store.search(query, scope, limit=1, mode="hybrid")

    assert vector[0].memory.id == distractor_memory.id
    assert hybrid[0].memory.id == exact_memory.id


async def test_subject_filter_excludes_other_subjects_and_overfetches(
    tmp_path: Path,
):
    query = "target"
    caroline_content = "Caroline has the filtered memory."
    melanie_content = "Melanie is the dense nearest neighbor."
    store = await _controlled_store(
        tmp_path,
        {
            query: [0.0, 1.0],
            caroline_content: [1.0, 0.0],
            melanie_content: [0.0, 1.0],
        },
    )
    scope = Scope(user_id="conversation")
    caroline = await store.add(
        caroline_content,
        scope,
        subject="Caroline",
    )
    await store.add(
        melanie_content,
        scope,
        subject="Melanie",
    )

    results = await store.search(
        query,
        scope,
        limit=1,
        filters={"subject": "caroline"},
        mode="vector",
    )

    assert [result.memory.id for result in results] == [caroline.id]


async def test_fts_memories_stays_synced_on_add_update_delete(store):
    scope = Scope(user_id="alice")
    memory = await store.add(
        "Alice uses codename Zephyr.",
        scope,
        subject="Alice",
    )

    added = await store.search("Zephyr", scope, mode="bm25")
    assert [result.memory.id for result in added] == [memory.id]

    updated = await store.update(
        memory.id,
        "Alice uses codename Orchid.",
        scope,
    )
    assert updated is not None
    assert await store.search("Zephyr", scope, mode="bm25") == []
    orchid = await store.search("Orchid", scope, mode="bm25")
    assert [result.memory.id for result in orchid] == [memory.id]

    assert await store.delete(memory.id, scope) is True
    assert await store.search("Orchid", scope, mode="bm25") == []


async def test_fts_memories_stays_synced_for_extraction_actions(store):
    scope = Scope(user_id="alice")
    add_session = await store.ingest_session_turns(
        SessionInput(
            source_session_id="session_add",
            turns=[
                Turn(
                    speaker="Alice",
                    text="Alice uses codename Zephyr.",
                    source_turn_id="D1:1",
                )
            ],
        ),
        scope,
    )
    added = await store.write_extracted_memories(
        add_session.session_id,
        scope,
        [
            MemoryAction(
                kind=MemoryActionKind.ADD,
                content="Alice uses codename Zephyr.",
                subject="Alice",
                internal_turn_ids=add_session.turn_ids,
            )
        ],
    )
    memory = added[0]
    assert [result.memory.id for result in await store.search(
        "Zephyr",
        scope,
        mode="bm25",
    )] == [memory.id]

    update_session = await store.ingest_session_turns(
        SessionInput(
            source_session_id="session_update",
            turns=[
                Turn(
                    speaker="Alice",
                    text="Alice uses codename Orchid.",
                    source_turn_id="D2:1",
                )
            ],
        ),
        scope,
    )
    await store.write_extracted_memories(
        update_session.session_id,
        scope,
        [
            MemoryAction(
                kind=MemoryActionKind.UPDATE,
                target_id=memory.id,
                content="Alice uses codename Orchid.",
                internal_turn_ids=update_session.turn_ids,
            )
        ],
    )
    assert await store.search("Zephyr", scope, mode="bm25") == []
    assert [result.memory.id for result in await store.search(
        "Orchid",
        scope,
        mode="bm25",
    )] == [memory.id]

    delete_session = await store.ingest_session_turns(
        SessionInput(
            source_session_id="session_delete",
            turns=[
                Turn(
                    speaker="Alice",
                    text="Forget the codename.",
                    source_turn_id="D3:1",
                )
            ],
        ),
        scope,
    )
    await store.write_extracted_memories(
        delete_session.session_id,
        scope,
        [
            MemoryAction(
                kind=MemoryActionKind.DELETE,
                target_id=memory.id,
            )
        ],
    )
    assert await store.search("Orchid", scope, mode="bm25") == []


async def test_reconciliation_uses_vector_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    content = "Alice likes exact matches."
    store = await _controlled_store(tmp_path, {content: [1.0, 0.0]})
    scope = Scope(user_id="alice")
    memory = await store.add(content, scope)
    calls: list[dict] = []
    original_search = store.search

    async def recording_search(query, scope, limit=10, **kwargs):
        calls.append(kwargs)
        return await original_search(query, scope, limit, **kwargs)

    monkeypatch.setattr(store, "search", recording_search)
    reconciler = RecordingReconciler()
    service = MemoryService(
        store,
        FakeLLMClient(
            recorded={
                "facts": [
                    {
                        "content": content,
                        "subject": "Alice",
                        "supporting_turn_ids": ["D1:1"],
                    }
                ]
            }
        ),
        reconciler,
    )

    await service.ingest_session(
        SessionInput(
            source_session_id="session_1",
            turns=[
                Turn(
                    speaker="Alice",
                    text=content,
                    source_turn_id="D1:1",
                )
            ],
        ),
        scope,
    )

    assert calls == [{"mode": "vector"}]
    assert reconciler.similar_memories[0].memory.id == memory.id
    assert reconciler.similar_memories[0].score == pytest.approx(1.0)


async def test_recall_evidence_fuses_memories_and_turns(store):
    scope = Scope(user_id="conversation")
    session = await store.ingest_session_turns(
        SessionInput(
            source_session_id="session_1",
            turns=[
                Turn(
                    speaker="Alice",
                    text="Alice moved to Berlin.",
                    source_turn_id="D1:1",
                )
            ],
        ),
        scope,
    )
    memory = await store.add(
        "Alice lives in Berlin.",
        scope,
        subject="Alice",
        internal_turn_ids=session.turn_ids,
        source_session_id="session_1",
    )
    service = MemoryService(
        store,
        FakeLLMClient(recorded={"facts": []}),
        AlwaysAddReconciler(),
    )

    results = await service.recall_evidence("Berlin", scope, limit=10)

    assert {result.kind for result in results} == {"memory", "turn"}
    memory_hit = next(result for result in results if result.kind == "memory")
    turn_hit = next(result for result in results if result.kind == "turn")
    assert memory_hit.memory_id == memory.id
    assert memory_hit.source_turn_ids == ["D1:1"]
    assert turn_hit.source_turn_ids == ["D1:1"]


async def test_optional_reranker_reorders_hybrid_candidates(store):
    scope = Scope(user_id="alice")
    await store.add("Alice has a baseline memory.", scope)
    preferred = await store.add("Alice has the preferred memory.", scope)
    service = MemoryService(
        store,
        FakeLLMClient(recorded={"facts": []}),
        AlwaysAddReconciler(),
        reranker=StubReranker(),
    )

    results = await service.search("Alice memory", scope, limit=2)

    assert results[0].memory.id == preferred.id
    assert results[0].score == 10.0
