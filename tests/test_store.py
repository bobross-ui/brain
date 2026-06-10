from datetime import datetime

import pytest

from brain.embeddings import SentenceTransformerEmbedder
from brain.models import Memory, Scope, SessionInput, Turn


def _assert_iso8601(value: str) -> None:
    datetime.fromisoformat(value)


async def test_add_returns_memory(store):
    scope = Scope(user_id="alice")

    memory = await store.add("Alice likes pasta.", scope)

    assert isinstance(memory, Memory)
    assert isinstance(memory.id, str)
    assert memory.id
    assert memory.content == "Alice likes pasta."
    assert memory.user_id == "alice"
    assert memory.namespace == "default"
    assert memory.content_hash
    _assert_iso8601(memory.created_at)
    _assert_iso8601(memory.updated_at)


async def test_search_scope_isolation(store):
    alice = Scope(user_id="alice")
    bob = Scope(user_id="bob")

    await store.add("Alice likes pasta.", alice)
    await store.add("Alice likes tiramisu.", alice)
    await store.add("Bob likes sushi.", bob)

    results = await store.search("likes", alice, limit=10)

    assert len(results) == 2
    assert all(result.memory.user_id == "alice" for result in results)


async def test_search_ordering(store):
    scope = Scope(user_id="alice")
    await store.add("aaa", scope)
    await store.add("aab", scope)
    await store.add("zzz", scope)

    results = await store.search("aaa", scope, limit=10)

    assert len(results) == 3
    assert all(
        results[index].score >= results[index + 1].score
        for index in range(len(results) - 1)
    )


async def test_get_found_and_not_found(store):
    alice = Scope(user_id="alice")
    bob = Scope(user_id="bob")
    memory = await store.add("Alice likes pasta.", alice)

    found = await store.get(memory.id, alice)
    missing = await store.get("not-a-real-id", alice)
    wrong_user = await store.get(memory.id, bob)

    assert found == memory
    assert missing is None
    assert wrong_user is None


async def test_delete(store):
    scope = Scope(user_id="alice")
    memory = await store.add("Alice likes pasta.", scope)

    assert await store.delete(memory.id, scope) is True
    assert await store.get(memory.id, scope) is None
    assert await store.delete("not-a-real-id", scope) is False


async def test_update(store):
    scope = Scope(user_id="alice")
    memory = await store.add("Alice likes pasta.", scope)

    updated = await store.update(memory.id, "Alice likes risotto.", scope)

    assert updated is not None
    assert updated.id == memory.id
    assert updated.content == "Alice likes risotto."
    assert updated.content_hash != memory.content_hash
    assert updated.updated_at >= updated.created_at

    fetched = await store.get(memory.id, scope)
    assert fetched == updated


async def test_add_and_update_provenance(store):
    scope = Scope(user_id="alice")
    first_session = await store.ingest_session_turns(
        SessionInput(
            source_session_id="session_1",
            turns=[
                Turn(
                    speaker="Alice",
                    text="I like pasta.",
                    source_turn_id="D1:1",
                )
            ],
        ),
        scope,
    )
    second_session = await store.ingest_session_turns(
        SessionInput(
            source_session_id="session_2",
            turns=[
                Turn(
                    speaker="Alice",
                    text="I prefer risotto.",
                    source_turn_id="D2:1",
                )
            ],
        ),
        scope,
    )

    memory = await store.add(
        "Alice likes pasta.",
        scope,
        subject="Alice",
        internal_turn_ids=first_session.turn_ids,
        observed_at="2026-06-01T00:00:00+00:00",
        source_session_id="session_1",
    )
    assert memory.subject == "Alice"
    assert memory.source_turn_ids == ["D1:1"]
    assert memory.source_session_id == "session_1"

    updated = await store.update(
        memory.id,
        "Alice prefers risotto.",
        scope,
        internal_turn_ids=second_session.turn_ids,
        observed_at="2026-06-02T00:00:00+00:00",
        source_session_id="session_2",
    )
    assert updated is not None
    assert updated.source_turn_ids == ["D2:1"]
    assert updated.source_session_id == "session_2"
    assert updated.observed_at == "2026-06-02T00:00:00+00:00"

    without_new_sources = await store.update(
        memory.id,
        "Alice still prefers risotto.",
        scope,
    )
    assert without_new_sources is not None
    assert without_new_sources.source_turn_ids == ["D2:1"]


async def test_namespace_isolation(store):
    default_scope = Scope(user_id="alice")
    work_scope = Scope(user_id="alice", namespace="work")

    await store.add("Alice likes pasta.", default_scope)
    await store.add("Alice likes sprint planning.", work_scope)

    results = await store.search("Alice likes", work_scope, limit=10)

    assert len(results) == 1
    assert results[0].memory.namespace == "work"
    assert results[0].memory.content == "Alice likes sprint planning."


@pytest.mark.slow
async def test_live_embedder_smoke():
    embedder = SentenceTransformerEmbedder()

    vec = await embedder.embed("hello world")

    assert len(vec) == 384
    assert isinstance(vec[0], float)
