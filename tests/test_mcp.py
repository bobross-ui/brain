import importlib
import sys
from types import ModuleType

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from brain.llm import FakeLLMClient
from brain.memory import MemoryService
from brain.reconcile import AlwaysAddReconciler


def _import_test_server(monkeypatch: pytest.MonkeyPatch, service: MemoryService) -> ModuleType:
    import brain.memory

    monkeypatch.setattr(brain.memory, "build_memory", lambda: service)
    sys.modules.pop("brain.mcp_server", None)
    return importlib.import_module("brain.mcp_server")


@pytest.fixture
def mcp_server(monkeypatch: pytest.MonkeyPatch, store) -> ModuleType:
    service = MemoryService(
        store,
        FakeLLMClient(
            recorded={
                "facts": [
                    {
                        "content": "The user loves hiking in the mountains.",
                        "subject": "user",
                        "supporting_turn_ids": ["message-0"],
                    }
                ]
            },
        ),
        AlwaysAddReconciler(),
    )
    module = _import_test_server(monkeypatch, service)
    yield module
    sys.modules.pop("brain.mcp_server", None)


async def test_remember_then_recall_round_trip(mcp_server: ModuleType):
    async with create_connected_server_and_client_session(
        mcp_server.mcp,
        raise_exceptions=True,
    ) as client:
        remember_result = await client.call_tool(
            "remember",
            {
                "messages": [
                    {"role": "user", "content": "I love hiking in the mountains."},
                    {"role": "assistant", "content": "That sounds wonderful!"},
                ],
                "user_id": "test-user",
            },
        )
        stored = remember_result.structuredContent["result"]
        assert isinstance(stored, list)
        assert len(stored) >= 1
        for memory in stored:
            assert "id" in memory
            assert "content" in memory
            assert isinstance(memory["id"], str)
            assert memory["subject"] == "user"
            assert memory["source_turn_ids"] == ["message-0"]

        recall_result = await client.call_tool(
            "recall",
            {
                "query": "outdoor activities",
                "user_id": "test-user",
            },
        )
        hits = recall_result.structuredContent["result"]
        assert isinstance(hits, list)
        assert len(hits) >= 1
        hit = hits[0]
        assert "memory" in hit
        assert "score" in hit
        assert isinstance(hit["score"], float)
        assert isinstance(hit["memory"]["content"], str)
        assert len(hit["memory"]["content"]) > 0
        assert hit["memory"]["subject"] == "user"
        assert hit["memory"]["source_turn_ids"] == ["message-0"]

        filtered_result = await client.call_tool(
            "recall",
            {
                "query": "outdoor activities",
                "user_id": "test-user",
                "filters": {"subject": "someone-else"},
            },
        )
        assert filtered_result.structuredContent["result"] == []

        evidence_result = await client.call_tool(
            "recall_evidence",
            {
                "query": "hiking mountains",
                "user_id": "test-user",
            },
        )
        evidence = evidence_result.structuredContent["result"]
        assert {item["kind"] for item in evidence} == {"memory", "turn"}


async def test_forget(mcp_server: ModuleType):
    async with create_connected_server_and_client_session(
        mcp_server.mcp,
        raise_exceptions=True,
    ) as client:
        remember_result = await client.call_tool(
            "remember",
            {
                "messages": [
                    {"role": "user", "content": "I enjoy reading novels."},
                ],
                "user_id": "test-user-forget",
            },
        )
        stored = remember_result.structuredContent["result"]
        assert len(stored) >= 1
        memory_id = stored[0]["id"]

        forget_result = await client.call_tool(
            "forget",
            {"id": memory_id, "user_id": "test-user-forget"},
        )
        outcome = forget_result.structuredContent
        assert outcome["deleted"] is True
        assert outcome["id"] == memory_id
