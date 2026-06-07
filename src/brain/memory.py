from brain.config import settings
from brain.embeddings import SentenceTransformerEmbedder
from brain.extract import Extractor
from brain.llm import LLMClient, OllamaLLMClient
from brain.models import (
    Memory,
    MemoryAction,
    MemoryActionKind,
    Reconciler,
    Scope,
    ScoredMemory,
)
from brain.reconcile import AlwaysAddReconciler
from brain.store.base import MemoryStore
from brain.store.sqlite import SQLiteMemoryStore, _apply_schema


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
            similar = await self._store.search(
                candidate.content,
                scope,
                limit=self._search_k,
            )
            action: MemoryAction = await self._reconciler.reconcile(
                candidate,
                similar,
            )
            if action.kind == MemoryActionKind.ADD:
                memory = await self._store.add(
                    action.content,
                    scope,
                    action.metadata,
                )
                stored.append(memory)
            elif action.kind == MemoryActionKind.UPDATE:
                await self._store.update(action.target_id, action.content, scope)
            elif action.kind == MemoryActionKind.DELETE:
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
    if settings.llm_provider != "ollama":
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    _apply_schema(settings.brain_db_path)
    embedder = SentenceTransformerEmbedder(settings.brain_embedder_model)
    store = SQLiteMemoryStore(settings.brain_db_path, embedder)
    llm = OllamaLLMClient(settings.llm_model)
    reconciler = AlwaysAddReconciler()
    return MemoryService(store, llm, reconciler)
