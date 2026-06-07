import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from brain.embeddings import FakeEmbedder
from brain.llm import FakeLLMClient
from brain.memory import MemoryService, build_memory
from brain.models import Scope
from brain.reconcile import AlwaysAddReconciler
from brain.store.sqlite import SQLiteMemoryStore


ROOT = Path(__file__).resolve().parents[1]
CONVERSATION_PATH = ROOT / "tests" / "fixtures" / "conversations" / "conv_01.json"
RECORDED_RESPONSE_PATH = (
    ROOT / "tests" / "fixtures" / "llm_responses" / "extract_conv_01.json"
)
EXPECTED_FACTS = {
    "The user moved to Berlin last month.",
    "The user works as a software engineer.",
    "The user cycles to work every day.",
    "The user plays chess in the evenings.",
    "The user is vegetarian.",
}


def _load_json(path: Path):
    return json.loads(path.read_text())


async def _run_fixture() -> None:
    messages = _load_json(CONVERSATION_PATH)
    recorded = _load_json(RECORDED_RESPONSE_PATH)

    with tempfile.TemporaryDirectory(prefix="brain-layer2-") as tmp_dir:
        db_path = str(Path(tmp_dir) / "demo.db")
        store = await SQLiteMemoryStore.create(db_path, FakeEmbedder())
        service = MemoryService(
            store,
            FakeLLMClient(recorded=recorded),
            AlwaysAddReconciler(),
        )
        stored = await service.add(messages, Scope(user_id="demo-user"))

    stored_contents = {memory.content for memory in stored}
    assert stored_contents == EXPECTED_FACTS

    print("Stored memories:")
    for index, memory in enumerate(stored, start=1):
        print(f'  {index}. "{memory.content}"')


async def _run_live() -> None:
    messages = _load_json(CONVERSATION_PATH)
    service = build_memory()
    stored = await service.add(messages, Scope(user_id="demo-user"))

    assert len(stored) >= 1
    assert isinstance(stored[0].content, str)
    assert stored[0].content

    print(f"Stored {len(stored)} live memory/memories:")
    for index, memory in enumerate(stored, start=1):
        print(f'  {index}. "{memory.content}"')


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("fixture", "live"),
        default="fixture",
    )
    args = parser.parse_args()

    if args.mode == "fixture":
        await _run_fixture()
    else:
        await _run_live()


if __name__ == "__main__":
    asyncio.run(main())
