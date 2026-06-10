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


class Turn(BaseModel):
    speaker: str
    text: str
    source_turn_id: str | None = None
    observed_at: str | None = None


class StoredTurn(Turn):
    id: str
    session_id: str
    seq: int
    ingested_at: str
    user_id: str
    namespace: str = "default"


class SessionInput(BaseModel):
    turns: list[Turn]
    source_session_id: str | None = None
    observed_at: str | None = None
    speaker_roster: dict | None = None
    metadata: dict = Field(default_factory=dict)


class IngestResult(BaseModel):
    session_id: str
    source_session_id: str
    turn_ids: list[str]
    memories: list[Memory] = Field(default_factory=list)
    extraction_completed_at: str | None = None
    extraction_skipped: bool = False


class RetrievedEvidence(BaseModel):
    kind: str
    content: str
    score: float
    memory_id: str | None = None
    turn_id: str | None = None
    source_turn_ids: list[str] = Field(default_factory=list)
    source_session_id: str | None = None
    observed_at: str | None = None
    event_date: str | None = None
    event_date_start: str | None = None
    event_date_end: str | None = None


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
