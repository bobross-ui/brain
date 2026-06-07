from typing import Any

from mcp.server.fastmcp import FastMCP

from brain.memory import build_memory
from brain.models import Scope


mcp = FastMCP("brain-memory")
memory = build_memory()


@mcp.tool()
async def remember(
    messages: list[dict[str, str]],
    user_id: str,
    agent_id: str | None = None,
    namespace: str = "default",
) -> list[dict[str, Any]]:
    scope = Scope(user_id=user_id, agent_id=agent_id, namespace=namespace)
    memories = await memory.add(messages=messages, scope=scope)
    return [m.model_dump() for m in memories]


@mcp.tool()
async def recall(
    query: str,
    user_id: str,
    agent_id: str | None = None,
    namespace: str = "default",
    limit: int = 10,
) -> list[dict[str, Any]]:
    scope = Scope(user_id=user_id, agent_id=agent_id, namespace=namespace)
    results = await memory.search(query=query, scope=scope, limit=limit)
    return [r.model_dump() for r in results]


@mcp.tool()
async def forget(
    id: str,
    user_id: str,
    agent_id: str | None = None,
    namespace: str = "default",
) -> dict[str, bool | str]:
    scope = Scope(user_id=user_id, agent_id=agent_id, namespace=namespace)
    result = await memory.forget(id=id, scope=scope)
    return {"deleted": result, "id": id}


if __name__ == "__main__":
    mcp.run()
