import hashlib
import json
import uuid

from brain.config import settings
from brain.embeddings import SentenceTransformerEmbedder
from brain.extract import Extractor
from brain.llm import LLMClient, build_llm_client
from brain.models import (
    FactCandidate,
    IngestResult,
    Memory,
    MemoryAction,
    MemoryActionKind,
    Reconciler,
    RetrievedEvidence,
    Scope,
    ScoredMemory,
    SessionInput,
    Turn,
)
from brain.reconcile import LLMReconciler
from brain.retrieval import (
    CrossEncoderReranker,
    Reranker,
    candidate_limit,
    reciprocal_rank_fusion,
    rerank,
)
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
        reranker: Reranker | None = None,
    ):
        self._store = store
        self._llm = llm
        self._reconciler = reconciler
        self._search_k = search_k
        self._reranker = reranker
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

        candidates = await self._extractor.extract(
            normalized.turns,
            roster=normalized.speaker_roster,
            session_observed_at=normalized.observed_at,
        )
        turn_context = {
            turn.source_turn_id: (internal_id, turn)
            for turn, internal_id in zip(
                normalized.turns,
                ingested.turn_ids,
                strict=True,
            )
            if turn.source_turn_id is not None
        }
        actions: list[MemoryAction] = []
        for candidate in candidates:
            similar = [
                scored
                for scored in await self._store.search(
                    candidate.content,
                    scope,
                    limit=self._search_k,
                    mode="vector",
                )
                if scored.score >= RECONCILE_SCORE_THRESHOLD
            ]
            action: MemoryAction = await self._reconciler.reconcile(candidate, similar)
            actions.append(
                self._attach_provenance(
                    action,
                    candidate,
                    turn_context=turn_context,
                    source_session_id=normalized.source_session_id,
                    session_observed_at=normalized.observed_at,
                )
            )

        memories = await self._store.write_extracted_memories(
            ingested.session_id,
            scope,
            actions,
        )
        return ingested.model_copy(update={"memories": memories})

    def _attach_provenance(
        self,
        action: MemoryAction,
        candidate: FactCandidate,
        *,
        turn_context: dict[str, tuple[str, Turn]],
        source_session_id: str | None,
        session_observed_at: str | None,
    ) -> MemoryAction:
        resolved = [
            turn_context[source_turn_id]
            for source_turn_id in candidate.source_turn_ids
            if source_turn_id in turn_context
        ]
        internal_turn_ids = list(dict.fromkeys(item[0] for item in resolved))
        unresolved_count = len(candidate.source_turn_ids) - len(resolved)

        subject = candidate.subject
        if subject is None and resolved:
            subject = resolved[0][1].speaker

        observed_at = session_observed_at
        if observed_at is None:
            observed_at = next(
                (
                    turn.observed_at
                    for _, turn in resolved
                    if turn.observed_at is not None
                ),
                None,
            )

        metadata = dict(candidate.metadata)
        metadata.update(action.metadata)
        metadata["unresolved_source_turn_ids"] = unresolved_count
        if action.kind == MemoryActionKind.ADD and not internal_turn_ids:
            metadata["unsourced"] = True
        elif action.kind == MemoryActionKind.UPDATE and not internal_turn_ids:
            metadata["provenance_update_skipped"] = True

        action_internal_turn_ids: list[str] | None = internal_turn_ids
        if action.kind == MemoryActionKind.UPDATE and not internal_turn_ids:
            action_internal_turn_ids = None

        return action.model_copy(
            update={
                "metadata": metadata,
                "subject": subject,
                "internal_turn_ids": action_internal_turn_ids,
                "source_session_id": source_session_id,
                "observed_at": observed_at,
            }
        )

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
        *,
        filters: dict | None = None,
        mode: str = "hybrid",
    ) -> list[ScoredMemory]:
        if limit <= 0:
            return []

        store_limit = candidate_limit(
            limit,
            overfetch=self._reranker is not None,
        )
        results = await self._store.search(
            query,
            scope,
            store_limit,
            filters=filters,
            mode=mode,
        )
        reranked = await rerank(
            query,
            results,
            [item.memory.content for item in results],
            self._reranker,
        )
        if reranked is None:
            return results[:limit]
        return [
            ScoredMemory(memory=item.memory, score=score)
            for item, score in reranked[:limit]
        ]

    async def recall_evidence(
        self,
        query: str,
        scope: Scope,
        limit: int = 10,
        *,
        filters: dict | None = None,
    ) -> list[RetrievedEvidence]:
        if limit <= 0:
            return []

        pool_limit = candidate_limit(limit, overfetch=True)
        memory_hits = await self._store.search(
            query,
            scope,
            pool_limit,
            filters=filters,
            mode="hybrid",
        )
        turn_hits = await self._store.search_turns(query, scope, pool_limit)
        memory_evidence = [
            RetrievedEvidence(
                kind="memory",
                content=hit.memory.content,
                score=hit.score,
                memory_id=hit.memory.id,
                source_turn_ids=hit.memory.source_turn_ids,
                source_session_id=hit.memory.source_session_id,
                observed_at=hit.memory.observed_at,
            )
            for hit in memory_hits
        ]
        evidence_by_id = {
            **{
                f"memory:{hit.memory_id}": hit
                for hit in memory_evidence
                if hit.memory_id is not None
            },
            **{
                f"turn:{hit.turn_id}": hit
                for hit in turn_hits
                if hit.turn_id is not None
            },
        }
        memory_ids = [
            f"memory:{hit.memory_id}"
            for hit in memory_evidence
            if hit.memory_id is not None
        ]
        turn_ids = [
            f"turn:{hit.turn_id}"
            for hit in turn_hits
            if hit.turn_id is not None
        ]
        fused_scores = reciprocal_rank_fusion([memory_ids, turn_ids])
        fused_ids = sorted(
            fused_scores,
            key=lambda item_id: fused_scores[item_id],
            reverse=True,
        )
        fused = [
            evidence_by_id[item_id].model_copy(
                update={"score": fused_scores[item_id]}
            )
            for item_id in fused_ids
        ]
        reranked = await rerank(
            query,
            fused,
            [item.content for item in fused],
            self._reranker,
        )
        if reranked is None:
            return fused[:limit]
        return [
            item.model_copy(update={"score": score})
            for item, score in reranked[:limit]
        ]

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
    reranker = (
        CrossEncoderReranker(settings.brain_reranker_model)
        if settings.brain_reranker_model
        else None
    )
    return MemoryService(
        store,
        llm,
        reconciler,
        search_k=5,
        reranker=reranker,
    )
