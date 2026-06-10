import importlib.util
import json
from pathlib import Path

from brain.models import RetrievedEvidence, Scope

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "eval_locomo", ROOT / "scripts" / "eval_locomo.py"
)
eval_locomo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_locomo)


class _RecordingService:
    def __init__(self):
        self.sessions = []

    async def ingest_session(self, session, scope):
        self.sessions.append(session)
        return None

    async def search(self, query, scope, limit):
        return []

    async def search_turns(self, query, scope, limit):
        return []


def _scored(content: str, source_turn_ids: list[str] | None = None):
    # Minimal stand-in matching the .memory.content access path used by the harness.
    class _M:
        def __init__(self, c, turn_ids):
            self.score = 0.9
            self.memory = type(
                "Mem",
                (),
                {
                    "id": "memory-1",
                    "content": c,
                    "source_turn_ids": turn_ids,
                    "source_session_id": "session_1",
                    "observed_at": "2023-05-07T10:00:00",
                },
            )

    return _M(content, source_turn_ids or [])


def test_session_keys_sorted_numerically():
    conversation = {
        "session_2": [],
        "session_10": [],
        "session_1": [],
        "speaker_a": "Alice",
    }
    assert eval_locomo._session_keys(conversation) == [
        "session_1",
        "session_2",
        "session_10",
    ]


def test_is_abstention():
    assert eval_locomo._is_abstention("No information available.")
    assert eval_locomo._is_abstention("That is not mentioned in the conversation")
    assert not eval_locomo._is_abstention("She moved to Berlin in 2021")


def test_recall_at_k_proxy():
    retrieved = [_scored("User moved to Berlin in 2021"), _scored("User likes jazz")]
    assert eval_locomo.recall_at_k("Berlin", retrieved) is True
    assert eval_locomo.recall_at_k("Tokyo", retrieved) is False
    # abstention answers are not scored for recall
    assert eval_locomo.recall_at_k("No information available", retrieved) is None


def test_recall_at_k_proxy_accepts_turn_evidence():
    retrieved = [
        RetrievedEvidence(
            kind="turn",
            content="Alice moved to Berlin in 2021",
            score=1.0,
            turn_id="turn-1",
            source_turn_ids=["D1:1"],
        )
    ]

    assert eval_locomo.recall_at_k("Berlin", retrieved) is True


def test_evidence_recall_scores_full_partial_and_missing_evidence():
    assert eval_locomo.evidence_recall(
        ["D1:3"],
        {"D1:3", "D2:1"},
    ) == 1.0
    assert eval_locomo.evidence_recall(
        ["D1:3", "D1:4"],
        {"D1:3"},
    ) == 0.5
    assert eval_locomo.evidence_recall([], {"D1:3"}) is None


def test_evidence_fields_flatten_memory_provenance_and_report_hit():
    retrieved = [
        _scored("Alice moved to Berlin.", ["D1:3", "D1:4"]),
        _scored("Alice likes jazz.", ["D2:1"]),
    ]

    fields = eval_locomo._evidence_fields(["D1:3", "D1:4"], retrieved)

    assert fields == {
        "gold_evidence": ["D1:3", "D1:4"],
        "retrieved_turn_ids": ["D1:3", "D1:4", "D2:1"],
        "evidence_recall": 1.0,
        "evidence_hit": True,
    }


def test_evidence_hit_is_none_without_gold_evidence():
    assert eval_locomo.evidence_hit([], {"D1:3"}) is None


def test_summary_evidence_recall_falls_back_to_proxy_for_old_records():
    assert eval_locomo._summary_evidence_recall({"recall_hit": True}) is True
    assert (
        eval_locomo._summary_evidence_recall(
            {"recall_hit": True, "evidence_recall": None}
        )
        is None
    )


def test_heuristic_correct():
    assert eval_locomo.heuristic_correct("Berlin", "She moved to Berlin") is True
    assert eval_locomo.heuristic_correct("Berlin", "She moved to Tokyo") is False
    # abstention: correct only if prediction also abstains
    assert eval_locomo.heuristic_correct("Not mentioned", "No information available") is True
    assert eval_locomo.heuristic_correct("Not mentioned", "She likes jazz") is False


def test_as_bool_coerces_unenforced_schema_values():
    # real booleans (ollama enforces the schema)
    assert eval_locomo._as_bool(True) is True
    assert eval_locomo._as_bool(False) is False
    # string values (DeepSeek does not enforce the schema) — the dangerous case
    assert eval_locomo._as_bool("false") is False
    assert eval_locomo._as_bool("False") is False
    assert eval_locomo._as_bool("true") is True
    assert eval_locomo._as_bool(None) is False


def test_gold_answer_falls_back_to_adversarial():
    assert eval_locomo._gold_answer({"answer": "x"}) == "x"
    assert eval_locomo._gold_answer({"adversarial_answer": "y"}) == "y"
    assert eval_locomo._gold_answer({}) == ""


def test_load_locomo_unwraps_dict(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"data": [{"sample_id": "a"}]}))
    assert eval_locomo.load_locomo(path) == [{"sample_id": "a"}]


async def test_ingest_conversation_orders_sessions_and_preserves_speakers():
    sample = {
        "sample_id": "conv-x",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_2_date_time": "2pm",
            "session_2": [{"speaker": "Alice", "text": "second"}],
            "session_1_date_time": "1pm",
            "session_1": [
                {"speaker": "Alice", "text": "first", "dia_id": "D1:1"},
                {"speaker": "Bob", "text": "hi", "dia_id": "D1:2"},
            ],
        },
    }
    service = _RecordingService()
    count = await eval_locomo.ingest_conversation(
        service, sample, Scope(user_id="conv-x"), max_sessions=0
    )

    assert count == 2
    # session_1 ingested before session_2
    first_session = service.sessions[0]
    assert first_session.source_session_id == "session_1"
    assert first_session.observed_at == "1pm"
    assert first_session.speaker_roster == {"speaker_a": "Alice", "speaker_b": "Bob"}
    assert first_session.turns[0].speaker == "Alice"
    assert first_session.turns[0].text == "first"
    assert first_session.turns[0].source_turn_id == "D1:1"
    assert first_session.turns[0].observed_at == "1pm"
    assert first_session.turns[1].speaker == "Bob"
    assert first_session.turns[1].source_turn_id == "D1:2"
    assert service.sessions[1].source_session_id == "session_2"
    assert service.sessions[1].turns[0].speaker == "Alice"
    assert service.sessions[1].turns[0].text == "second"
