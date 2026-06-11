from brain.llm import FakeLLMClient
from brain.memory import MemoryService
from brain.models import Scope, SessionInput, Turn
from brain.reconcile import AlwaysAddReconciler
from brain.temporal import resolve


def test_resolve_relative_dates_and_ranges_against_anchor():
    yesterday = resolve("Caroline went yesterday.", "2023-05-08")
    two_days = resolve("Caroline went two days ago.", "2023-05-08")
    last_week = resolve("Caroline went last week.", "2023-05-08")

    assert yesterday is not None
    assert yesterday.date == "2023-05-07"
    assert yesterday.raw_phrase == "yesterday"
    assert two_days is not None
    assert two_days.date == "2023-05-06"
    assert last_week is not None
    assert last_week.start == "2023-05-01"
    assert last_week.end == "2023-05-07"


def test_resolve_absolute_locomo_date_and_unparseable_text():
    explicit = resolve("The event was on 7 May 2023.")

    assert explicit is not None
    assert explicit.date == "2023-05-07"
    assert explicit.raw_phrase == "7 May 2023"
    assert resolve("The event happened sometime recently.") is None


def test_relative_query_without_anchor_is_not_guessed():
    assert resolve("What happened yesterday?") is None


async def test_ingest_normalizes_relative_date_and_keeps_raw_phrase(store):
    scope = Scope(user_id="conversation")
    content = "Caroline attended a support group yesterday."
    service = MemoryService(
        store,
        FakeLLMClient(
            recorded={
                "facts": [
                    {
                        "content": content,
                        "subject": "Caroline",
                        "supporting_turn_ids": ["D1:1"],
                    }
                ]
            }
        ),
        AlwaysAddReconciler(),
    )

    result = await service.ingest_session(
        SessionInput(
            source_session_id="session_1",
            observed_at="1:56 pm on 8 May, 2023",
            turns=[
                Turn(
                    speaker="Caroline",
                    text=content,
                    source_turn_id="D1:1",
                )
            ],
        ),
        scope,
    )

    memory = result.memories[0]
    assert memory.content == content
    assert memory.event_date == "2023-05-07"
    assert memory.event_date_start is None
    assert memory.event_date_end is None
    assert memory.metadata["raw_time_phrase"] == "yesterday"
    assert memory.metadata["temporal_method"] == "relative_yesterday"


async def test_event_on_filter_matches_points_and_ranges(store):
    scope = Scope(user_id="conversation")
    point = await store.add(
        "Caroline attended a support group.",
        scope,
        event_date="2023-05-07",
    )
    span = await store.add(
        "Caroline attended events throughout the week.",
        scope,
        event_date_start="2023-05-01",
        event_date_end="2023-05-07",
    )
    await store.add(
        "Caroline attended a later event.",
        scope,
        event_date="2023-05-08",
    )

    results = await store.search(
        "Caroline attended",
        scope,
        limit=10,
        filters={"event_on": "2023-05-07"},
        mode="vector",
    )

    assert {result.memory.id for result in results} == {point.id, span.id}


async def test_recall_evidence_exposes_normalized_dates(store):
    scope = Scope(user_id="conversation")
    memory = await store.add(
        "Caroline attended a support group yesterday.",
        scope,
        event_date="2023-05-07",
    )
    service = MemoryService(
        store,
        FakeLLMClient(recorded={"facts": []}),
        AlwaysAddReconciler(),
    )

    results = await service.recall_evidence(
        "support group",
        scope,
        limit=10,
    )

    hit = next(item for item in results if item.memory_id == memory.id)
    assert hit.event_date == "2023-05-07"
