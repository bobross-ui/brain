import importlib.util
import json
from pathlib import Path

from brain.models import Scope

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "eval_locomo", ROOT / "scripts" / "eval_locomo.py"
)
eval_locomo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_locomo)


class _RecordingService:
    def __init__(self):
        self.adds: list[list[dict]] = []

    async def add(self, messages, scope):
        self.adds.append(messages)
        return []

    async def search(self, query, scope, limit):
        return []


def _scored(content: str):
    # Minimal stand-in matching the .memory.content access path used by the harness.
    class _M:
        def __init__(self, c):
            self.memory = type("Mem", (), {"content": c})

    return _M(content)


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
                {"speaker": "Alice", "text": "first"},
                {"speaker": "Bob", "text": "hi"},
            ],
        },
    }
    service = _RecordingService()
    count = await eval_locomo.ingest_conversation(
        service, sample, Scope(user_id="conv-x"), max_sessions=0
    )

    assert count == 2
    # session_1 ingested before session_2
    first_session = service.adds[0]
    assert first_session[0] == {"role": "system", "content": "[Conversation session dated 1pm]"}
    assert {"role": "Alice", "content": "first"} in first_session
    assert {"role": "Bob", "content": "hi"} in first_session
    assert service.adds[1][-1] == {"role": "Alice", "content": "second"}
