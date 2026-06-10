from brain.llm import LLMClient
from brain.models import FactCandidate, Turn


SYSTEM_PROMPT = (
    "You are a memory extraction assistant. Given a conversation between named speakers, "
    "extract all distinct atomic facts worth remembering. Each fact must be a single, "
    "self-contained statement that names the person or entity it is about. Cite only the "
    "provided turn IDs that directly support the fact. Output only JSON matching the "
    "provided schema."
)

USER_PROMPT_TEMPLATE = (
    "Extract speaker- and entity-aware atomic facts from the conversation.\n"
    "Speaker roster: {roster}\n"
    "Session observed at: {session_observed_at}\n\n"
    "Conversation (each line starts with its source turn ID):\n{conversation_text}"
)

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "subject": {"type": ["string", "null"]},
                    "supporting_turn_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["content", "subject", "supporting_turn_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}


class Extractor:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    async def extract(
        self,
        turns: list[Turn],
        *,
        roster: dict | None = None,
        session_observed_at: str | None = None,
    ) -> list[FactCandidate]:
        conversation_text = "\n".join(
            _format_turn(turn, index) for index, turn in enumerate(turns)
        )
        llm_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    roster=roster or {},
                    session_observed_at=session_observed_at or "(unknown)",
                    conversation_text=conversation_text,
                ),
            },
        ]
        response = await self._llm.chat_json(
            llm_messages,
            schema=EXTRACTION_SCHEMA,
            temperature=0.0,
        )
        candidates: list[FactCandidate] = []
        for fact in response["facts"]:
            if isinstance(fact, str):
                # Tolerate pre-Phase-2 recorded responses without weakening the
                # structured schema requested from current LLM backends.
                content = fact
                subject = None
                source_turn_ids: list[str] = []
            else:
                content = str(fact["content"])
                subject = fact.get("subject")
                source_turn_ids = [
                    str(turn_id) for turn_id in fact.get("supporting_turn_ids", [])
                ]

            candidates.append(
                FactCandidate(
                    content=content,
                    subject=str(subject) if subject is not None else None,
                    source_turn_ids=source_turn_ids,
                    metadata={
                        "source": "extraction",
                        "source_turn_ids": source_turn_ids,
                    },
                )
            )
        return candidates


def _format_turn(turn: Turn, index: int) -> str:
    source_turn_id = turn.source_turn_id or f"turn-{index}"
    observed_at = f" @ {turn.observed_at}" if turn.observed_at else ""
    return f"[{source_turn_id}]{observed_at} {turn.speaker}: {turn.text}"
