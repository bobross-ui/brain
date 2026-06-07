from brain.models import (
    FactCandidate,
    MemoryAction,
    MemoryActionKind,
    Reconciler,
    ScoredMemory,
)


class AlwaysAddReconciler(Reconciler):
    async def reconcile(
        self,
        candidate: FactCandidate,
        similar_memories: list[ScoredMemory],
    ) -> MemoryAction:
        return MemoryAction(
            kind=MemoryActionKind.ADD,
            content=candidate.content,
            metadata=candidate.metadata,
        )
