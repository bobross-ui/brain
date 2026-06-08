import json

import httpx
import pytest

from brain.llm import DeepSeekLLMClient


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    captured: list[dict] = []
    responses: list[dict] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, headers=None, json=None):
        _FakeAsyncClient.captured.append({"url": url, "headers": headers, "json": json})
        return _FakeResponse(_FakeAsyncClient.responses.pop(0))


def _content(payload: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


@pytest.fixture
def fake_http(monkeypatch):
    _FakeAsyncClient.captured = []
    _FakeAsyncClient.responses = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    return _FakeAsyncClient


async def test_chat_json_parses_content_and_builds_request(fake_http):
    fake_http.responses = [_content({"facts": ["likes pizza"]})]
    client = DeepSeekLLMClient("deepseek-v4-pro", api_key="sk-test")
    schema = {"type": "object", "properties": {"facts": {"type": "array"}}}
    original = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

    result = await client.chat_json(original, schema, temperature=0.0)

    assert result == {"facts": ["likes pizza"]}
    sent = fake_http.captured[0]["json"]
    assert sent["model"] == "deepseek-v4-pro"
    assert sent["response_format"] == {"type": "json_object"}
    assert "max_tokens" in sent
    assert fake_http.captured[0]["headers"]["Authorization"] == "Bearer sk-test"
    # schema + the literal word "json" must be injected into the final prompt message
    last_message = sent["messages"][-1]["content"]
    assert "json" in last_message.lower()
    assert json.dumps(schema) in last_message
    # the caller's messages must not be mutated by schema injection
    assert original == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


async def test_chat_json_retries_once_on_empty_content(fake_http):
    fake_http.responses = [
        {"choices": [{"message": {"content": ""}}]},
        _content({"action": "ADD"}),
    ]
    client = DeepSeekLLMClient("deepseek-v4-pro", api_key="sk-test")

    result = await client.chat_json([{"role": "user", "content": "x"}], {"type": "object"})

    assert result == {"action": "ADD"}
    assert len(fake_http.captured) == 2


async def test_chat_json_raises_after_two_empty_responses(fake_http):
    fake_http.responses = [
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": ""}}]},
    ]
    client = DeepSeekLLMClient("deepseek-v4-pro", api_key="sk-test")

    with pytest.raises(RuntimeError, match="empty content"):
        await client.chat_json([{"role": "user", "content": "x"}], {"type": "object"})


def test_missing_api_key_raises():
    with pytest.raises(ValueError, match="API key"):
        DeepSeekLLMClient("deepseek-v4-pro", api_key="")
