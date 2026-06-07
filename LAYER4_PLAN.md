# Brain — Layer 4: MCP server — Implementation Spec

> Autonomous build spec. Self-contained — implement exactly what's here; depend on no other file.

---

## 0. Scope

**Build (in scope):**
- `src/brain/mcp_server.py` — FastMCP server with three tools: `remember`, `recall`, `forget`
- `tests/test_mcp.py` — in-process round-trip test asserting cross-call persistence
- Add `mcp==1.27.2` to `pyproject.toml` dependencies

**Do NOT build (out of scope):**
- Auth of any kind
- HTTP/SSE transport or any non-stdio transport
- New memory logic — this layer is a transport adapter only
- Changes to L1–L3 (store, embedder, extractor, reconciler, MemoryService)
- Multi-tenant API-key routing

---

## 1. Setup — do this FIRST

```bash
uv add "mcp==1.27.2"
```

Verify the addition appears in `pyproject.toml` under `[project.dependencies]` and that `uv.lock` is updated.

**Files created or modified this layer:**

| File | Action |
|---|---|
| `pyproject.toml` | add `mcp==1.27.2` |
| `src/brain/mcp_server.py` | create |
| `tests/test_mcp.py` | create |

**Run target (stdio launch, used by agents):**

```bash
uv run python -m brain.mcp_server
```

This must block on stdio — the process is the MCP server. No HTTP port opened.

---

## 2. Interfaces & data contracts

### 2.1 Pinned types from §5.1 (restate verbatim — change nothing)

```python
# src/brain/models.py — already implemented by L1–L3; DO NOT redefine
class Scope(BaseModel):
    user_id: str
    agent_id: str | None = None
    namespace: str = "default"

class Memory(BaseModel):
    id: str
    content: str
    user_id: str
    agent_id: str | None = None
    namespace: str = "default"
    metadata: dict = Field(default_factory=dict)
    content_hash: str
    created_at: str
    updated_at: str

class ScoredMemory(BaseModel):
    memory: Memory
    score: float

# src/brain/memory.py — already implemented by L3; DO NOT redefine
class MemoryService:
    async def add(self, messages: list[dict], scope: Scope) -> list[Memory]: ...
    async def search(self, query: str, scope: Scope, limit: int = 10) -> list[ScoredMemory]: ...
    async def get(self, id: str, scope: Scope) -> Memory | None: ...
    async def forget(self, id: str, scope: Scope) -> bool: ...

def build_memory() -> MemoryService: ...
```

### 2.2 MCP server registration

```python
# src/brain/mcp_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("brain-memory")

# ... tool definitions (§2.3) ...

if __name__ == "__main__":
    mcp.run()   # no args = stdio transport (default)
```

`mcp.run()` with no arguments selects stdio transport. No other transport is configured.

### 2.3 Tool schemas (flat typed args — scope NEVER nested on the wire)

Scope args (`user_id`, `agent_id`, `namespace`) are flat parameters. Each handler constructs `Scope(...)` internally. Handlers are async.

---

#### Tool: `remember`

**Input (flat):**

| Parameter | Type | Notes |
|---|---|---|
| `messages` | `list[dict[str, str]]` | OpenAI-style conversation, e.g. `[{"role": "user", "content": "..."}]` |
| `user_id` | `str` | required |
| `agent_id` | `str \| None` | optional, default `None` |
| `namespace` | `str` | default `"default"` |

**Handler body (exact call):**

```python
scope = Scope(user_id=user_id, agent_id=agent_id, namespace=namespace)
memories: list[Memory] = await memory.add(messages=messages, scope=scope)
return [m.model_dump() for m in memories]
```

**Output:** `list[dict]` — each dict is a serialised `Memory` (all fields from `Memory.model_dump()`).

**Wire result unwrap:** `result.structuredContent["result"]` (list-returning tool).

---

#### Tool: `recall`

**Input (flat):**

| Parameter | Type | Notes |
|---|---|---|
| `query` | `str` | semantic search query |
| `user_id` | `str` | required |
| `agent_id` | `str \| None` | optional, default `None` |
| `namespace` | `str` | default `"default"` |
| `limit` | `int` | default `10` |

**Handler body (exact call):**

```python
scope = Scope(user_id=user_id, agent_id=agent_id, namespace=namespace)
results: list[ScoredMemory] = await memory.search(query=query, scope=scope, limit=limit)
return [r.model_dump() for r in results]
```

**Output:** `list[dict]` — each dict is a serialised `ScoredMemory`, containing a nested `memory` dict and a `score` float.

**Wire result unwrap:** `result.structuredContent["result"]` (list-returning tool).

---

#### Tool: `forget`

**Input (flat):**

| Parameter | Type | Notes |
|---|---|---|
| `id` | `str` | Memory id — always a `str` (never a UUID object) |
| `user_id` | `str` | required |
| `agent_id` | `str \| None` | optional, default `None` |
| `namespace` | `str` | default `"default"` |

**Handler body (exact call):**

```python
scope = Scope(user_id=user_id, agent_id=agent_id, namespace=namespace)
result: bool = await memory.forget(id=id, scope=scope)
return {"deleted": result, "id": id}
```

**Output:** `dict` with keys `"deleted"` (bool) and `"id"` (str).

**Wire result unwrap:** `result.structuredContent` directly (dict-returning tool).

---

### 2.4 Singleton construction

The `MemoryService` is constructed once at module level via `build_memory()`:

```python
from brain.memory import build_memory

memory = build_memory()
```

All three tool handlers share this single instance. No per-request construction.

---

## 3. Implementation order

Each step is a working state; do not proceed to the next until the current step passes.

**Step 1 — dependency**

Run `uv add "mcp==1.27.2"`. Verify `python -c "import mcp; print(mcp.__version__)"` prints `1.27.2`.

**Step 2 — skeleton server (no tools yet)**

Create `src/brain/mcp_server.py` with the `FastMCP("brain-memory")` instance and `if __name__ == "__main__": mcp.run()`. Verify `uv run python -m brain.mcp_server` starts without error (it will block waiting for stdin; Ctrl-C to exit).

**Step 3 — `remember` tool**

Add the `remember` tool registration to `mcp_server.py` using the exact handler body from §2.3. Test manually: from an in-process session, call `remember` with a short fixture conversation and assert the return value is a non-empty list of dicts, each with an `"id"` key.

**Step 4 — `recall` tool**

Add the `recall` tool. Verify that after a `remember` call, a subsequent `recall` with a related query returns at least one result whose `"memory"."content"` is a non-empty string and whose `"score"` is a float.

**Step 5 — `forget` tool**

Add the `forget` tool. Verify: call `remember`, extract an `id` from the result, call `forget` with that `id`, assert `{"deleted": True, "id": <id>}`.

**Step 6 — test file**

Write `tests/test_mcp.py` (§4). The test must pass via `uv run pytest tests/test_mcp.py -v`.

**Step 7 — full suite**

Run `uv run pytest` (all layers). All tests must pass. Fix any import issues introduced by the new module.

---

## 4. Test fixtures

### 4.1 In-process client setup

Use `mcp.shared.memory.create_connected_server_and_client_session` with `raise_exceptions=True`. The `FastMCP` instance is imported from `brain.mcp_server`.

```python
# tests/test_mcp.py (structure — not literal code; implement exactly this logic)
import pytest
import asyncio
from mcp.shared.memory import create_connected_server_and_client_session
from brain.mcp_server import mcp   # the FastMCP instance

@pytest.mark.asyncio
async def test_remember_then_recall_round_trip():
    async with create_connected_server_and_client_session(
        mcp, raise_exceptions=True
    ) as (server, client):

        # Call 1: remember a conversation
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
        stored = remember_result.structuredContent["result"]   # list-returning tool
        assert isinstance(stored, list)
        assert len(stored) >= 1
        for m in stored:
            assert "id" in m
            assert "content" in m
            assert isinstance(m["id"], str)

        # Call 2: recall — SEPARATE call, asserts cross-call persistence
        recall_result = await client.call_tool(
            "recall",
            {
                "query": "outdoor activities",
                "user_id": "test-user",
            },
        )
        hits = recall_result.structuredContent["result"]       # list-returning tool
        assert isinstance(hits, list)
        assert len(hits) >= 1
        hit = hits[0]
        assert "memory" in hit
        assert "score" in hit
        assert isinstance(hit["score"], float)
        assert isinstance(hit["memory"]["content"], str)
        assert len(hit["memory"]["content"]) > 0
```

### 4.2 Forget test (supplementary)

```python
@pytest.mark.asyncio
async def test_forget():
    async with create_connected_server_and_client_session(
        mcp, raise_exceptions=True
    ) as (server, client):

        remember_result = await client.call_tool(
            "remember",
            {
                "messages": [{"role": "user", "content": "I enjoy reading novels."}],
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
        outcome = forget_result.structuredContent   # dict-returning tool — direct, no ["result"]
        assert outcome["deleted"] is True
        assert outcome["id"] == memory_id
```

### 4.3 Fixtures from prior layers

`test_mcp.py` may rely on the same temp-DB fixture from `tests/conftest.py` that L1–L3 use (the real sqlite-vec database, temp file). No DB mocking. The fake embedder may be used to keep the test fast and deterministic, but the in-process MCP client must go through the real tool dispatch path.

---

## 5. Pinned values and assertions

| Concern | Pinned value |
|---|---|
| MCP package | `mcp==1.27.2` (official MCP Python SDK; NOT `fastmcp`) |
| Transport | stdio via `FastMCP.run()` with no arguments |
| Server name | `"brain-memory"` |
| Tool names | `remember`, `recall`, `forget` (exact — no aliases) |
| Scope construction | `Scope(user_id=..., agent_id=..., namespace=...)` — never pass scope as a nested dict on the wire |
| `id` type | always `str` (never `uuid.UUID`) at every boundary |
| `limit` parameter | present on `recall`; passed through to `memory.search(..., limit=limit)` |
| `remember` output unwrap | `result.structuredContent["result"]` |
| `recall` output unwrap | `result.structuredContent["result"]` |
| `forget` output unwrap | `result.structuredContent` (direct, no `["result"]`) |
| Round-trip assertion | `recall` in a separate tool call returns ≥1 hit with `memory.content` non-empty and `score` is a float |
| Facade entry point | `build_memory()` from `brain.memory` — one call at module level; no per-request construction |
| Facade method names | `add`, `search`, `get`, `forget` — verbatim from §5.1 |
| Handler style | async (`async def`) for all three tools |

---

## 6. Layer completion verification

**Done when:** `uv run pytest tests/test_mcp.py -v`

Expected:
```
tests/test_mcp.py::test_remember_then_recall_round_trip PASSED
tests/test_mcp.py::test_forget PASSED
```

Both tests pass without mocking the DB or the MCP transport layer. The `recall` call in `test_remember_then_recall_round_trip` is a **separate** tool call from the `remember` call, proving that memories persist across calls within the same server session.

Additionally, verify the server starts cleanly:

```bash
uv run python -m brain.mcp_server
```

Expected: process blocks on stdin (no error output). Ctrl-C terminates cleanly.

**Hands off to:** the product — a runnable MCP memory server any agent can connect to via stdio by launching `uv run python -m brain.mcp_server` and speaking the MCP protocol.
