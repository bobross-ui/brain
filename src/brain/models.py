from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, Field


class Scope(BaseModel):
    user_id: str
    agent_id: str | None = None
    namespace: str = "default"


class Memory(BaseModel):
    id: str
    content: str
    user_id: str
    agent_id: str | None = None
    namespace: str = "default"
    metadata: dict = Field(default_factory=dict)
    content_hash: str
    created_at: str
    updated_at: str


class ScoredMemory(BaseModel):
    memory: Memory
    score: float


class FactCandidate(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)


class MemoryActionKind(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


class MemoryAction(BaseModel):
    kind: MemoryActionKind
    content: str | None = None
    target_id: str | None = None
    metadata: dict = Field(default_factory=dict)


class Reconciler(ABC):
    @abstractmethod
    async def reconcile(
        self,
        candidate: FactCandidate,
        similar_memories: list[ScoredMemory],
    ) -> MemoryAction:
        ...
