from abc import ABC, abstractmethod

from brain.models import Memory, Scope, ScoredMemory


class MemoryStore(ABC):
    @abstractmethod
    async def add(
        self,
        content: str,
        scope: Scope,
        metadata: dict | None = None,
    ) -> Memory:
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        scope: Scope,
        limit: int = 10,
    ) -> list[ScoredMemory]:
        ...

    @abstractmethod
    async def get(self, id: str, scope: Scope) -> Memory | None:
        ...

    @abstractmethod
    async def delete(self, id: str, scope: Scope) -> bool:
        ...

    @abstractmethod
    async def update(self, id: str, content: str, scope: Scope) -> Memory | None:
        ...
