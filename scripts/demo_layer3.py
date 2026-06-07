import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from brain.config import settings
from brain.embeddings import FakeEmbedder
from brain.llm import LLMClient
from brain.memory import MemoryService, build_memory
from brain.models import (
    FactCandidate,
    MemoryAction,
    Reconciler,
    Scope,
    ScoredMemory,
)
from brain.reconcile import LLMReconciler
from brain.store.sqlite import SQLiteMemoryStore


ROOT = Path(__file__).resolve().parents[1]
CONVERSATION_PATH = ROOT / "tests" / "fixtures" / "conversations" / "pizza_to_sushi.json"
RECORDED_RESPONSE_PATH = (
    ROOT / "tests" / "fixtures" / "llm_responses" / "reconcile_pizza_sushi.json"
)


class ReplayLLM(LLMClient):
    def __init__(self, responses: list[dict]):
        self._responses = list(responses)

    async def chat_json(
        self,
        messages: list[dict],
        schema: dict,
        temperature: float = 0.0,
    ) -> dict:
        if not self._responses:
            raise RuntimeError("ReplayLLM received more calls than expected")
        response = self._responses.pop(0)
        return response.get("response", response)


class RecordingReconciler(Reconciler):
    def __init__(self, inner: Reconciler):
        self._inner = inner
        self.actions: list[MemoryAction] = []

    async def reconcile(
        self,
        candidate: FactCandidate,
        similar_memories: list[ScoredMemory],
    ) -> MemoryAction:
        action = await self._inner.reconcile(candidate, similar_memories)
        self.actions.append(action)
        return action


def _load_json(path: Path):
    return json.loads(path.read_text())


async def _run_fixture() -> None:
    messages = _load_json(CONVERSATION_PATH)
    responses = _load_json(RECORDED_RESPONSE_PATH)
    scope = Scope(user_id="demo-layer3")

    with tempfile.TemporaryDirectory(prefix="brain-layer3-") as tmp_dir:
        db_path = str(Path(tmp_dir) / "demo.db")
        store = await SQLiteMemoryStore.create(db_path, FakeEmbedder())
        seed = await store.add("User likes pizza", scope)
        llm = ReplayLLM(responses)
        reconciler = RecordingReconciler(LLMReconciler(llm))
        service = MemoryService(store, llm, reconciler, search_k=5)

        await service.add(messages, scope)
        results = await service.search("food preference", scope, limit=10)

    assert len(reconciler.actions) == 1
    action = reconciler.actions[0]
    assert action.target_id == seed.id
    assert action.content == "User prefers sushi"
    assert len(results) == 1
    assert results[0].memory.id == seed.id
    assert "sushi" in results[0].memory.content.lower()
    assert all("pizza" not in result.memory.content.lower() for result in results)

    print(f'Seeded: "User likes pizza" (id={seed.id})')
    print('Fed: "Actually I prefer sushi now"')
    print(
        f"Action: {action.kind.value}  target_id={action.target_id}  "
        f'content="{action.content}"'
    )
    print("Final memories (1):")
    print(f'  [0] "{results[0].memory.content}"  score={results[0].score:.3f}')
    print("PASS: pizza updated to sushi in place, no duplicate.")


async def _run_live() -> None:
    messages = _load_json(CONVERSATION_PATH)
    scope = Scope(user_id="demo-layer3-live")

    with tempfile.TemporaryDirectory(prefix="brain-layer3-live-") as tmp_dir:
        previous_db_path = settings.brain_db_path
        settings.brain_db_path = str(Path(tmp_dir) / "demo.db")
        try:
            service = build_memory()
            seed = await service._store.add("User likes pizza", scope)
            returned = await service.add(messages, scope)
            results = await service.search("food preference", scope, limit=10)
        finally:
            settings.brain_db_path = previous_db_path

    assert len(returned) >= 1
    assert all(memory.user_id == scope.user_id for memory in returned)
    assert results

    print(f'Seeded: "User likes pizza" (id={seed.id})')
    print('Fed: "Actually I prefer sushi now"')
    print(f"Returned {len(returned)} memory/memories from live reconciliation.")
    print(f"Final memories ({len(results)}):")
    for index, result in enumerate(results):
        print(f'  [{index}] "{result.memory.content}"  score={result.score:.3f}')


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
