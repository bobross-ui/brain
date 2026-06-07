from brain.llm import LLMClient
from brain.models import (
    FactCandidate,
    MemoryAction,
    MemoryActionKind,
    Reconciler,
    ScoredMemory,
)


DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["ADD", "UPDATE", "DELETE", "NOOP"]},
        "target_index": {"type": ["integer", "null"]},
        "content": {"type": ["string", "null"]},
        "reason": {"type": ["string", "null"]},
    },
    "required": ["action"],
}

SYSTEM_PROMPT = """You are a memory reconciliation engine. Given a NEW FACT and a list of EXISTING MEMORIES,
decide what action to take. Respond ONLY with valid JSON matching the schema.

Actions:
- ADD: the fact is genuinely new; no existing memory covers it.
- UPDATE: the fact contradicts or supersedes an existing memory; update that memory in place.
- DELETE: the fact explicitly invalidates an existing memory with no replacement.
- NOOP: the fact is already captured by an existing memory; discard it.

Rules:
- Prefer UPDATE over ADD when the new fact and an existing memory are about the same topic
  but the new fact replaces the old value (e.g. food preference changed).
- target_index is the 0-based index into EXISTING MEMORIES (null if not applicable).
- content is the final text to store (for ADD or UPDATE); null for DELETE or NOOP.
"""

USER_PROMPT_TEMPLATE = """NEW FACT: {candidate_content}

EXISTING MEMORIES:
{existing_memories}"""


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


class LLMReconciler(Reconciler):
    def __init__(self, llm: LLMClient):
        self._llm = llm

    async def reconcile(
        self,
        candidate: FactCandidate,
        similar_memories: list[ScoredMemory],
    ) -> MemoryAction:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    candidate_content=candidate.content,
                    existing_memories=_format_existing_memories(similar_memories),
                ),
            },
        ]
        result = await self._llm.chat_json(
            messages,
            schema=DECISION_SCHEMA,
            temperature=0.0,
        )

        kind = _parse_action_kind(result.get("action"))
        content = result.get("content")
        if kind in (MemoryActionKind.ADD, MemoryActionKind.UPDATE) and not content:
            content = candidate.content
        if kind in (MemoryActionKind.DELETE, MemoryActionKind.NOOP):
            content = None

        if kind == MemoryActionKind.ADD:
            return MemoryAction(kind=kind, content=content)

        target_id = _resolve_target_id(result.get("target_index"), similar_memories)
        if target_id is None:
            return MemoryAction(
                kind=MemoryActionKind.ADD,
                content=content or candidate.content,
            )

        return MemoryAction(kind=kind, content=content, target_id=target_id)


def _format_existing_memories(similar_memories: list[ScoredMemory]) -> str:
    if not similar_memories:
        return "(none)"
    return "\n".join(
        f"{index}. {scored.memory.content}"
        for index, scored in enumerate(similar_memories)
    )


def _parse_action_kind(value: object) -> MemoryActionKind:
    try:
        return MemoryActionKind(value)
    except (TypeError, ValueError):
        return MemoryActionKind.ADD


def _resolve_target_id(
    target_index: object,
    similar_memories: list[ScoredMemory],
) -> str | None:
    if not isinstance(target_index, int):
        return None
    if target_index < 0 or target_index >= len(similar_memories):
        return None
    return similar_memories[target_index].memory.id
