import asyncio
import tempfile
from pathlib import Path

from brain.config import settings
from brain.embeddings import SentenceTransformerEmbedder
from brain.models import Scope
from brain.store.sqlite import SQLiteMemoryStore


ALICE_FACTS = [
    "Alice loves Italian food, especially pasta and risotto.",
    "Alice is allergic to shellfish.",
    "Alice's favourite dessert is tiramisu.",
    "Alice drinks oat milk, not cow's milk.",
    "Alice eats lunch at noon every weekday.",
]

BOB_FACTS = [
    "Bob prefers Japanese food and sushi.",
    "Bob is vegetarian.",
]


async def main() -> None:
    embedder = SentenceTransformerEmbedder(settings.brain_embedder_model)

    with tempfile.TemporaryDirectory(prefix="brain-layer1-") as tmp_dir:
        db_path = str(Path(tmp_dir) / "demo.db")
        store = await SQLiteMemoryStore.create(db_path, embedder)

        alice = Scope(user_id="alice")
        bob = Scope(user_id="bob")

        for fact in ALICE_FACTS:
            await store.add(fact, alice)
        for fact in BOB_FACTS:
            await store.add(fact, bob)

        query = "what food do I like"
        print(f'Querying as alice: "{query}"')
        print("Results:")
        alice_results = await store.search(query, alice, limit=5)
        for index, result in enumerate(alice_results, start=1):
            assert result.memory.user_id == "alice"
            print(
                f'  {index}. score={result.score:.3f}  "{result.memory.content}"'
            )

        assert all(
            alice_results[index].score >= alice_results[index + 1].score
            for index in range(len(alice_results) - 1)
        )

        bob_results = await store.search(query, bob, limit=5)
        assert len(bob_results) == 2
        assert all(result.memory.user_id == "bob" for result in bob_results)

        print()
        print(f"Bob's rows when searched as bob: {len(bob_results)} result(s)")
        print("Alice rows never appear in bob's search.")


if __name__ == "__main__":
    asyncio.run(main())
