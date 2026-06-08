from abc import ABC, abstractmethod
import json

import ollama


DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"


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


class DeepSeekLLMClient(LLMClient):
    """Calls the DeepSeek API (OpenAI-compatible) for JSON chat completions.

    DeepSeek's JSON mode is ``response_format={"type": "json_object"}`` and does
    NOT enforce a JSON schema, so the requested schema is appended to the final
    prompt message as guidance (this also satisfies DeepSeek's rule that the word
    "json" must appear in the prompt). DeepSeek occasionally returns empty content
    (a documented intermittent limitation), so an empty body is retried once.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = DEEPSEEK_DEFAULT_BASE_URL,
        *,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ):
        if not api_key:
            raise ValueError(
                "DeepSeek API key is required (set DEEPSEEK_API_KEY in the environment)"
            )
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._timeout = timeout

    async def chat_json(
        self,
        messages: list[dict],
        schema: dict,
        temperature: float = 0.0,
    ) -> dict:
        import httpx

        guided = [dict(message) for message in messages]
        schema_hint = (
            "\n\nReturn a single json object matching this JSON schema. "
            "Output only the json, with no markdown fences or commentary:\n"
            + json.dumps(schema)
        )
        if guided:
            guided[-1]["content"] = guided[-1].get("content", "") + schema_hint
        else:
            guided.append({"role": "user", "content": schema_hint})

        payload = {
            "model": self._model,
            "messages": guided,
            "temperature": temperature,
            "max_tokens": self._max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for _ in range(2):
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                if content:
                    return json.loads(content)
        raise RuntimeError(
            "DeepSeek returned empty content twice (known intermittent API issue)"
        )


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


def build_llm_client(provider: str, model: str) -> LLMClient:
    """Construct an LLMClient for the given provider.

    Supported providers: ``ollama`` (local) and ``deepseek`` (DeepSeek API).
    DeepSeek credentials are read from settings (``DEEPSEEK_API_KEY`` /
    ``DEEPSEEK_BASE_URL``).
    """
    if provider == "ollama":
        return OllamaLLMClient(model)
    if provider == "deepseek":
        from brain.config import settings

        return DeepSeekLLMClient(
            model,
            api_key=settings.deepseek_api_key or "",
            base_url=settings.deepseek_base_url,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")
