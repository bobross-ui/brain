from brain.config import settings
from brain.embeddings import SentenceTransformerEmbedder
from brain.extract import Extractor
from brain.llm import LLMClient, build_llm_client
from brain.models import (
    Memory,
    MemoryAction,
    MemoryActionKind,
    Reconciler,
    Scope,
    ScoredMemory,
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

    async def add(self, messages: list[dict], scope: Scope) -> list[Memory]:
        candidates = await self._extractor.extract(messages)
        stored: list[Memory] = []
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
            if action.kind == MemoryActionKind.ADD and action.content:
                memory = await self._store.add(action.content, scope, action.metadata)
                stored.append(memory)
            elif (
                action.kind == MemoryActionKind.UPDATE
                and action.target_id
                and action.content
            ):
                memory = await self._store.update(
                    action.target_id,
                    action.content,
                    scope,
                )
                if memory is not None:
                    stored.append(memory)
            elif action.kind == MemoryActionKind.DELETE and action.target_id:
                await self._store.delete(action.target_id, scope)
            elif action.kind == MemoryActionKind.NOOP:
                pass

        return stored

    async def search(
        self,
        query: str,
        scope: Scope,
        limit: int = 10,
    ) -> list[ScoredMemory]:
        return await self._store.search(query, scope, limit)

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
