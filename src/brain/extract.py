from brain.llm import LLMClient
from brain.models import FactCandidate


SYSTEM_PROMPT = (
    "You are a memory extraction assistant. Given a conversation, extract all distinct "
    "atomic facts about the user. Each fact must be a single, self-contained statement. "
    "Output only JSON matching the provided schema."
)

USER_PROMPT_TEMPLATE = (
    "Extract all atomic facts about the user from the following conversation.\n\n"
    "Conversation:\n{conversation_text}"
)

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}


class Extractor:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    async def extract(self, messages: list[dict]) -> list[FactCandidate]:
        conversation_text = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        llm_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    conversation_text=conversation_text,
                ),
            },
        ]
        response = await self._llm.chat_json(
            llm_messages,
            schema=EXTRACTION_SCHEMA,
            temperature=0.0,
        )
        return [
            FactCandidate(content=fact, metadata={"source": "extraction"})
            for fact in response["facts"]
        ]
