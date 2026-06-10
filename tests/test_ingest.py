import sqlite3

from brain.llm import LLMClient
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
from brain.store.sqlite import _open_db


class SequencedLLM(LLMClient):
    def __init__(self, responses: list[dict | Exception]):
        self._responses = list(responses)
        self.calls = 0

    async def chat_json(
        self,
        messages: list[dict],
        schema: dict,
        temperature: float = 0.0,
    ) -> dict:
        self.calls += 1
        if not self._responses:
            raise AssertionError("SequencedLLM received more calls than expected")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class TargetUpdateReconciler(Reconciler):
    def __init__(self, target_id: str):
        self._target_id = target_id

    async def reconcile(
        self,
        candidate: FactCandidate,
        similar_memories: list[ScoredMemory],
    ) -> MemoryAction:
        return MemoryAction(
            kind=MemoryActionKind.UPDATE,
            target_id=self._target_id,
            content=candidate.content,
        )


def _fact(
    content: str,
    *,
    subject: str | None = "Alice",
    source_turn_ids: list[str] | None = None,
) -> dict:
    return {
        "content": content,
        "subject": subject,
        "supporting_turn_ids": source_turn_ids or ["D1:1"],
    }


def _db(store) -> sqlite3.Connection:
    return _open_db(store._db_path)


def _count(store, table: str) -> int:
    db = _db(store)
    try:
        return int(db.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
    finally:
        db.close()


def _memory_count(store, scope: Scope) -> int:
    db = _db(store)
    try:
        return int(
            db.execute(
                "SELECT count(*) FROM memories WHERE user_id = ? AND namespace = ?",
                (scope.user_id, scope.namespace),
            ).fetchone()[0]
        )
    finally:
        db.close()


async def test_ingest_session_turns_preserves_raw_fields(store):
    scope = Scope(user_id="conv-1")
    session = SessionInput(
        source_session_id="session_1",
        observed_at="2023-05-07T10:00:00",
        speaker_roster={"speaker_a": "Caroline", "speaker_b": "Melanie"},
        turns=[
            Turn(
                speaker="Caroline",
                text="I moved to Berlin.",
                source_turn_id="D1:1",
            ),
            Turn(
                speaker="Melanie",
                text="That sounds exciting.",
                source_turn_id="D1:2",
                observed_at="2023-05-07T10:03:00",
            ),
        ],
    )

    result = await store.ingest_session_turns(session, scope)

    assert len(result.turn_ids) == 2
    first = await store.get_turn(result.turn_ids[0], scope)
    second = await store.get_turn(result.turn_ids[1], scope)
    assert first is not None
    assert second is not None
    assert first.session_id == result.session_id
    assert first.source_turn_id == "D1:1"
    assert first.speaker == "Caroline"
    assert first.observed_at == "2023-05-07T10:00:00"
    assert first.text == "I moved to Berlin."
    assert second.source_turn_id == "D1:2"
    assert second.speaker == "Melanie"
    assert second.observed_at == "2023-05-07T10:03:00"
    assert second.text == "That sounds exciting."


async def test_search_turns_returns_source_turn_id(store):
    scope = Scope(user_id="conv-1")
    await store.ingest_session_turns(
        SessionInput(
            source_session_id="session_1",
            turns=[
                Turn(speaker="Alice", text="Alice moved to Berlin.", source_turn_id="D1:1"),
                Turn(speaker="Bob", text="Bob likes Tokyo.", source_turn_id="D1:2"),
            ],
        ),
        scope,
    )

    results = await store.search_turns("Berlin", scope, limit=10)

    assert len(results) == 1
    assert results[0].kind == "turn"
    assert results[0].content == "Alice moved to Berlin."
    assert results[0].source_turn_ids == ["D1:1"]
    assert results[0].source_session_id == "session_1"


async def test_reingest_reuses_session_and_does_not_duplicate_turns_or_memories(store):
    scope = Scope(user_id="conv-1")
    service = MemoryService(
        store,
        SequencedLLM([{"facts": [_fact("Alice moved to Berlin.")]}]),
        AlwaysAddReconciler(),
    )
    session = SessionInput(
        source_session_id="session_1",
        turns=[Turn(speaker="Alice", text="Alice moved to Berlin.", source_turn_id="D1:1")],
    )

    first = await service.ingest_session(session, scope)
    second = await service.ingest_session(session, scope)

    assert first.session_id == second.session_id
    assert second.extraction_skipped is True
    assert _count(store, "sessions") == 1
    assert _count(store, "turns") == 1
    assert _count(store, "fts_turns") == 1
    assert _memory_count(store, scope) == 1


async def test_adapter_allow_duplicates_inserts_second_copy(store):
    scope = Scope(user_id="adapter-user")
    service = MemoryService(
        store,
        SequencedLLM(
            [
                {
                    "facts": [
                        _fact(
                            "The user likes Berlin.",
                            subject="user",
                            source_turn_ids=["message-0"],
                        )
                    ]
                },
                {
                    "facts": [
                        _fact(
                            "The user likes Berlin.",
                            subject="user",
                            source_turn_ids=["message-0"],
                        )
                    ]
                },
            ]
        ),
        AlwaysAddReconciler(),
    )
    messages = [{"role": "user", "content": "I like Berlin."}]

    await service.add(messages, scope, allow_duplicates=True)
    await service.add(messages, scope, allow_duplicates=True)

    assert _count(store, "sessions") == 2
    assert _count(store, "turns") == 2
    assert _memory_count(store, scope) == 2


async def test_allow_duplicates_does_not_rekey_explicit_turn_ids(store):
    scope = Scope(user_id="conv-1")
    service = MemoryService(
        store,
        SequencedLLM([{"facts": [_fact("Alice moved to Berlin.")]}]),
        AlwaysAddReconciler(),
    )
    session = SessionInput(
        source_session_id="session_1",
        turns=[Turn(speaker="Alice", text="Alice moved to Berlin.", source_turn_id="D1:1")],
    )

    await service.ingest_session(session, scope)
    second = await service.ingest_session(session, scope, allow_duplicates=True)

    assert second.extraction_skipped is True
    assert _count(store, "sessions") == 1
    assert _count(store, "turns") == 1
    assert _memory_count(store, scope) == 1


async def test_retry_after_failed_extraction_reuses_turns_and_clears_partials(store):
    scope = Scope(user_id="conv-1")
    service = MemoryService(
        store,
        SequencedLLM(
            [RuntimeError("boom"), {"facts": [_fact("Alice moved to Berlin.")]}]
        ),
        AlwaysAddReconciler(),
    )
    session = SessionInput(
        source_session_id="session_1",
        turns=[Turn(speaker="Alice", text="Alice moved to Berlin.", source_turn_id="D1:1")],
    )

    try:
        await service.ingest_session(session, scope)
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("Expected extraction failure")

    db = _db(store)
    try:
        session_row = db.execute(
            "SELECT id, extraction_completed_at FROM sessions"
        ).fetchone()
        assert session_row["extraction_completed_at"] is None
        session_id = session_row["id"]
    finally:
        db.close()

    await store.add(
        "partial memory from failed attempt",
        scope,
        {"ingest_session_id": session_id},
    )

    result = await service.ingest_session(session, scope)

    assert result.extraction_skipped is False
    assert _count(store, "sessions") == 1
    assert _count(store, "turns") == 1
    assert _count(store, "fts_turns") == 1
    assert _memory_count(store, scope) == 1
    assert result.memories[0].content == "Alice moved to Berlin."


async def test_fts_does_not_grow_on_reingest(store):
    scope = Scope(user_id="conv-1")
    session = SessionInput(
        source_session_id="session_1",
        turns=[Turn(speaker="Alice", text="Alice moved to Berlin.", source_turn_id="D1:1")],
    )

    await store.ingest_session_turns(session, scope)
    await store.ingest_session_turns(session, scope)
    results = await store.search_turns("Berlin", scope, limit=10)

    assert _count(store, "turns") == 1
    assert _count(store, "fts_turns") == 1
    assert len(results) == 1


async def test_ingest_links_memory_to_source_turn_and_surfaces_context(store):
    scope = Scope(user_id="conv-1")
    service = MemoryService(
        store,
        SequencedLLM(
            [
                {
                    "facts": [
                        _fact(
                            "Caroline moved to Berlin.",
                            subject="Caroline",
                            source_turn_ids=["D1:1"],
                        )
                    ]
                }
            ]
        ),
        AlwaysAddReconciler(),
    )
    session = SessionInput(
        source_session_id="session_1",
        observed_at="2023-05-07T10:00:00",
        speaker_roster={"speaker_a": "Caroline", "speaker_b": "Melanie"},
        turns=[
            Turn(
                speaker="Caroline",
                text="I moved to Berlin.",
                source_turn_id="D1:1",
            ),
            Turn(
                speaker="Melanie",
                text="That sounds exciting.",
                source_turn_id="D1:2",
            ),
        ],
    )

    result = await service.ingest_session(session, scope)

    assert len(result.memories) == 1
    memory = result.memories[0]
    assert memory.subject == "Caroline"
    assert memory.source_turn_ids == ["D1:1"]
    assert memory.source_session_id == "session_1"
    assert memory.observed_at == "2023-05-07T10:00:00"
    assert memory.metadata["unresolved_source_turn_ids"] == 0

    db = _db(store)
    try:
        link = db.execute(
            """
            SELECT memory_sources.turn_id, turns.source_turn_id
            FROM memory_sources
            JOIN turns ON turns.id = memory_sources.turn_id
            WHERE memory_sources.memory_id = ?
            """,
            (memory.id,),
        ).fetchone()
        assert link["turn_id"] == result.turn_ids[0]
        assert link["source_turn_id"] == "D1:1"
    finally:
        db.close()


async def test_resolver_drops_unknown_and_cross_session_turn_ids(store):
    scope = Scope(user_id="conv-1")
    other = await store.ingest_session_turns(
        SessionInput(
            source_session_id="other_session",
            turns=[
                Turn(
                    speaker="Caroline",
                    text="A turn from another session.",
                    source_turn_id="D9:9",
                )
            ],
        ),
        scope,
    )
    service = MemoryService(
        store,
        SequencedLLM(
            [
                {
                    "facts": [
                        _fact(
                            "Caroline moved to Berlin.",
                            subject="Caroline",
                            source_turn_ids=["D1:1", "D9:9"],
                        ),
                        _fact(
                            "Caroline likes an unknown place.",
                            subject=None,
                            source_turn_ids=["D8:8"],
                        ),
                    ]
                }
            ]
        ),
        AlwaysAddReconciler(),
    )

    result = await service.ingest_session(
        SessionInput(
            source_session_id="session_1",
            turns=[
                Turn(
                    speaker="Caroline",
                    text="I moved to Berlin.",
                    source_turn_id="D1:1",
                )
            ],
        ),
        scope,
    )

    linked, unsourced = result.memories
    assert linked.source_turn_ids == ["D1:1"]
    assert linked.metadata["unresolved_source_turn_ids"] == 1
    assert other.turn_ids[0] not in result.turn_ids
    assert unsourced.source_turn_ids == []
    assert unsourced.metadata["unresolved_source_turn_ids"] == 1
    assert unsourced.metadata["unsourced"] is True


async def test_update_replaces_provenance_and_empty_resolution_preserves_links(store):
    scope = Scope(user_id="conv-1")
    original_session = await store.ingest_session_turns(
        SessionInput(
            source_session_id="session_0",
            turns=[
                Turn(
                    speaker="Alice",
                    text="I like pizza.",
                    source_turn_id="D0:1",
                )
            ],
        ),
        scope,
    )
    memory = await store.add(
        "Alice likes pizza.",
        scope,
        subject="Alice",
        internal_turn_ids=original_session.turn_ids,
        source_session_id="session_0",
    )
    service = MemoryService(
        store,
        SequencedLLM(
            [
                {
                    "facts": [
                        _fact(
                            "Alice prefers sushi.",
                            source_turn_ids=["D1:1"],
                        )
                    ]
                },
                {
                    "facts": [
                        _fact(
                            "Alice prefers ramen.",
                            source_turn_ids=["D2:9"],
                        )
                    ]
                },
            ]
        ),
        TargetUpdateReconciler(memory.id),
    )

    first = await service.ingest_session(
        SessionInput(
            source_session_id="session_1",
            turns=[
                Turn(
                    speaker="Alice",
                    text="I prefer sushi.",
                    source_turn_id="D1:1",
                )
            ],
        ),
        scope,
    )
    assert first.memories[0].source_turn_ids == ["D1:1"]
    assert first.memories[0].source_session_id == "session_1"

    second = await service.ingest_session(
        SessionInput(
            source_session_id="session_2",
            turns=[
                Turn(
                    speaker="Alice",
                    text="I prefer ramen.",
                    source_turn_id="D2:1",
                )
            ],
        ),
        scope,
    )

    updated = second.memories[0]
    assert updated.content == "Alice prefers ramen."
    assert updated.source_turn_ids == ["D1:1"]
    assert updated.metadata["unresolved_source_turn_ids"] == 1
    assert updated.metadata["provenance_update_skipped"] is True
