from abc import ABC, abstractmethod
import json

import ollama


class LLMClient(ABC):
    @abstractmethod
    async def chat_json(
        self,
        messages: list[dict],
        schema: dict,
        temperature: float = 0.0,
    ) -> dict:
        ...


class OllamaLLMClient(LLMClient):
    def __init__(self, model: str):
        self._model = model

    async def chat_json(
        self,
        messages: list[dict],
        schema: dict,
        temperature: float = 0.0,
    ) -> dict:
        client = ollama.AsyncClient()
        response = await client.chat(
            model=self._model,
            messages=messages,
            format=schema,
            options={"temperature": temperature},
        )
        return json.loads(response.message.content)


class FakeLLMClient(LLMClient):
    """Returns a pre-recorded dict. Used in deterministic tests."""

    def __init__(self, recorded: dict):
        self._recorded = recorded

    async def chat_json(
        self,
        messages: list[dict],
        schema: dict,
        temperature: float = 0.0,
    ) -> dict:
        return self._recorded
