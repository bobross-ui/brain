import hashlib
import json
import uuid

from brain.config import settings
from brain.embeddings import SentenceTransformerEmbedder
from brain.extract import Extractor
from brain.llm import LLMClient, build_llm_client
from brain.models import (
    IngestResult,
    Memory,
    MemoryAction,
    Reconciler,
    RetrievedEvidence,
    Scope,
    ScoredMemory,
    SessionInput,
    Turn,
)
from brain.reconcile import LLMReconciler
from brain.store.base import MemoryStore
from brain.store.sqlite import SQLiteMemoryStore, _apply_schema


RECONCILE_SCORE_THRESHOLD = 0.3


class MemoryService:
    def __init__(
        self,
        store: MemoryStore,
        llm: LLMClient,
        reconciler: Reconciler,
        *,
        search_k: int = 5,
    ):
        self._store = store
        self._llm = llm
        self._reconciler = reconciler
        self._search_k = search_k
        self._extractor = Extractor(llm)

    async def add(
        self,
        messages: list[dict],
        scope: Scope,
        *,
        allow_duplicates: bool = False,
    ) -> list[Memory]:
        session = self._session_from_messages(messages)
        result = await self.ingest_session(
            session,
            scope,
            allow_duplicates=allow_duplicates,
            _adapter_path=True,
        )
        return result.memories

    async def ingest_session(
        self,
        session: SessionInput,
        scope: Scope,
        *,
        allow_duplicates: bool = False,
        _adapter_path: bool = False,
    ) -> IngestResult:
        normalized = self._normalize_session(
            session,
            allow_duplicates=allow_duplicates,
            adapter_path=_adapter_path,
        )
        ingested = await self._store.ingest_session_turns(normalized, scope)
        if ingested.extraction_completed_at is not None:
            ingested.extraction_skipped = True
            return ingested

        messages = [
            {"role": turn.speaker, "content": turn.text}
            for turn in normalized.turns
        ]
        candidates = await self._extractor.extract(messages)
        actions: list[MemoryAction] = []
        for candidate in candidates:
            similar = [
                scored
                for scored in await self._store.search(
                    candidate.content,
                    scope,
                    limit=self._search_k,
                )
                if scored.score >= RECONCILE_SCORE_THRESHOLD
            ]
            action: MemoryAction = await self._reconciler.reconcile(candidate, similar)
            actions.append(action)

        memories = await self._store.write_extracted_memories(
            ingested.session_id,
            scope,
            actions,
        )
        return ingested.model_copy(update={"memories": memories})

    def _session_from_messages(
        self,
        messages: list[dict],
    ) -> SessionInput:
        normalized_messages = [
            {
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            }
            for message in messages
        ]
        digest = hashlib.sha256(
            json.dumps(
                normalized_messages,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        source_session_id = f"add:{digest}"

        turns = []
        for index, message in enumerate(normalized_messages):
            source_turn_id = f"message-{index}"
            turns.append(
                Turn(
                    speaker=message["role"],
                    text=message["content"],
                    source_turn_id=source_turn_id,
                )
            )
        return SessionInput(turns=turns, source_session_id=source_session_id)

    def _normalize_session(
        self,
        session: SessionInput,
        *,
        allow_duplicates: bool,
        adapter_path: bool,
    ) -> SessionInput:
        source_session_id = session.source_session_id
        if source_session_id is None:
            source_session_id = self._stable_session_id(session)

        nonce = uuid.uuid4().hex if adapter_path and allow_duplicates else None
        if nonce is not None and not source_session_id.endswith(f":{nonce}"):
            source_session_id = f"{source_session_id}:{nonce}"

        turns = []
        for index, turn in enumerate(session.turns):
            source_turn_id = turn.source_turn_id or f"turn-{index}"
            if nonce is not None and not source_turn_id.endswith(f":{nonce}"):
                source_turn_id = f"{source_turn_id}:{nonce}"
            turns.append(turn.model_copy(update={"source_turn_id": source_turn_id}))

        return session.model_copy(
            update={
                "source_session_id": source_session_id,
                "turns": turns,
            }
        )

    def _stable_session_id(self, session: SessionInput) -> str:
        payload = {
            "observed_at": session.observed_at,
            "speaker_roster": session.speaker_roster,
            "turns": [
                {
                    "speaker": turn.speaker,
                    "text": turn.text,
                    "source_turn_id": turn.source_turn_id,
                    "observed_at": turn.observed_at,
                }
                for turn in session.turns
            ],
        }
        digest = hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()[:16]
        return f"session:{digest}"

    async def search(
        self,
        query: str,
        scope: Scope,
        limit: int = 10,
    ) -> list[ScoredMemory]:
        return await self._store.search(query, scope, limit)

    async def search_turns(
        self,
        query: str,
        scope: Scope,
        limit: int = 10,
    ) -> list[RetrievedEvidence]:
        return await self._store.search_turns(query, scope, limit)

    async def get(self, id: str, scope: Scope) -> Memory | None:
        return await self._store.get(id, scope)

    async def forget(self, id: str, scope: Scope) -> bool:
        return await self._store.delete(id, scope)


def build_memory() -> MemoryService:
    # Build the LLM client first so an unsupported provider or missing DeepSeek
    # key fails before any database side effects.
    llm = build_llm_client(settings.llm_provider, settings.llm_model)
    embedder = SentenceTransformerEmbedder(settings.brain_embedder_model)
    _apply_schema(settings.brain_db_path, embedder.dim)
    store = SQLiteMemoryStore(settings.brain_db_path, embedder)
    reconciler = LLMReconciler(llm)
    return MemoryService(store, llm, reconciler, search_k=5)
