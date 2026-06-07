from brain.models import Memory, Scope, ScoredMemory
from brain.store.base import MemoryStore


class MemoryService:
    def __init__(self, store: MemoryStore, llm, reconciler, *, search_k: int = 5):
        raise NotImplementedError("MemoryService is implemented in Layer 2/3.")

    async def add(self, messages: list[dict], scope: Scope) -> list[Memory]:
        raise NotImplementedError("MemoryService.add is implemented in Layer 2.")

    async def search(
        self,
        query: str,
        scope: Scope,
        limit: int = 10,
    ) -> list[ScoredMemory]:
        raise NotImplementedError("MemoryService.search is implemented in Layer 2/3.")

    async def get(self, id: str, scope: Scope) -> Memory | None:
        raise NotImplementedError("MemoryService.get is implemented in Layer 2/3.")

    async def forget(self, id: str, scope: Scope) -> bool:
        raise NotImplementedError("MemoryService.forget is implemented in Layer 2/3.")


def build_memory() -> MemoryService:
    raise NotImplementedError("build_memory is implemented in Layer 3/4.")
