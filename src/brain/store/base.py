from abc import ABC, abstractmethod

from brain.models import (
    IngestResult,
    Memory,
    MemoryAction,
    RetrievedEvidence,
    Scope,
    ScoredMemory,
    SessionInput,
    StoredTurn,
)


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

    async def ingest_session_turns(
        self,
        session: SessionInput,
        scope: Scope,
    ) -> IngestResult:
        raise NotImplementedError

    async def write_extracted_memories(
        self,
        session_id: str,
        scope: Scope,
        actions: list[MemoryAction],
    ) -> list[Memory]:
        raise NotImplementedError

    async def get_turn(self, id: str, scope: Scope) -> StoredTurn | None:
        raise NotImplementedError

    async def search_turns(
        self,
        query: str,
        scope: Scope,
        limit: int = 10,
    ) -> list[RetrievedEvidence]:
        raise NotImplementedError
