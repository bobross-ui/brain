# Brain — Layer 3: Reconciliation (ADD/UPDATE/DELETE/NOOP) — Implementation Spec

> Autonomous build spec. Self-contained — implement exactly what's here; depend on no other file.
> §5.1 of IMPLEMENTATION_PLAN.md is authoritative for all shared types; they are restated verbatim below.

---

## 0. Scope

**Build (in scope):**
- `LLMReconciler(Reconciler)` in `src/brain/reconcile.py` — a near-pure decision: receives `(candidate: FactCandidate, similar_memories: list[ScoredMemory])`, calls `LLMClient.chat_json`, returns `MemoryAction`.
- Wire `LLMReconciler` as the active reconciler: pass it to `MemoryService.__init__` as the `reconciler` arg, replacing `AlwaysAddReconciler`. The write-loop itself is NOT touched.
- Add `MemoryService.search`, `MemoryService.get`, `MemoryService.forget` pass-through methods (L2 shipped only `add`).
- Add `build_memory()` factory function in `src/brain/memory.py`.
- Recorded action fixture: `tests/fixtures/llm_responses/reconcile_pizza_sushi.json`.
- `tests/test_reconcile.py`: unit tests for UPDATE/DELETE/NOOP via injected candidate lists + `StubLLM`; one deterministic integration test (pizza → sushi UPDATE in place); one live smoke test (structural only).
- `scripts/demo_layer3.py`: seeds "User likes pizza", feeds "Actually I prefer sushi now", asserts UPDATED in place.

**Do NOT build (out of scope):**
- MCP server (`mcp_server.py`) — that is Layer 4.
- Any change to L1's store internals (`store/base.py`, `store/sqlite.py`, `store/schema.sql`).
- Any change to L2's extraction step (`extract.py`) or the write-loop in `memory.py`.
- Any change to `llm.py` — use `LLMClient.chat_json` as-is.
- Any new `MemoryStore` method — `update` already exists from L1.
- Advanced retrieval (hybrid search, reranking, recency/decay).
- Background/async reconciliation queue.
- Storage-layer `content_hash` uniqueness rejection.
- HTTP/SSE MCP transport or auth.

---

## 1. Setup — do this FIRST

No new dependencies. Everything required already exists:

- `LLMClient.chat_json` — the only LLM call interface; default impl is `OllamaLLMClient` pinned to `settings.llm_model` (`qwen2.5:3b-instruct`).
- `MemoryStore.search` — scope-filtered cosine KNN, returns `list[ScoredMemory]`; used by the write-loop to fetch candidates.
- `MemoryStore.update` — re-embeds content + bumps `updated_at`; implemented and unit-tested in L1; L3 is the first write path to call it.
- `MemoryStore.delete` — implemented in L1.
- All shared types (`Memory`, `ScoredMemory`, `FactCandidate`, `MemoryAction`, `MemoryActionKind`, `Reconciler`, `Scope`) — in `src/brain/models.py`; restated verbatim in §2 below.

Verify Ollama is running locally with `qwen2.5:3b-instruct` pulled before running live paths:

```
ollama pull qwen2.5:3b-instruct
```

No changes to `pyproject.toml` are expected. If `StubLLM` or test utilities require `pytest-asyncio`, confirm it is already present from L2.

---

## 2. Interfaces & data contracts

### 2.1 Shared types (§5.1 VERBATIM — change NOTHING)

```python
# ---------- src/brain/models.py ----------

class Scope(BaseModel):                 # the scope value object — pass THIS, not loose kwargs or a dict
    user_id: str                        # REQUIRED — never global
    agent_id: str | None = None         # carried for forward-compat; see v0 note below
    namespace: str = "default"

class Memory(BaseModel):                # the stored record (the `embedding` lives in the store, not on this object)
    id: str
    content: str
    user_id: str
    agent_id: str | None = None
    namespace: str = "default"
    metadata: dict = Field(default_factory=dict)
    content_hash: str
    created_at: str                     # ISO-8601 UTC text
    updated_at: str                     # ISO-8601 UTC text

class ScoredMemory(BaseModel):          # search result — NESTED: access .memory.id and .score
    memory: Memory
    score: float                        # cosine similarity, higher-is-better, [-1, 1]

class FactCandidate(BaseModel):         # L2 extraction output
    content: str
    metadata: dict = Field(default_factory=dict)

class MemoryActionKind(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"

class MemoryAction(BaseModel):          # STRUCTURED from L2 onward — there is NO bare-Enum stage
    kind: MemoryActionKind
    content: str | None = None          # set for ADD / UPDATE
    target_id: str | None = None        # a Memory.id (str); set for UPDATE / DELETE
    metadata: dict = Field(default_factory=dict)

class Reconciler(ABC):                  # the reconciliation seam (lives in models.py)
    @abstractmethod
    async def reconcile(self, candidate: FactCandidate,
                        similar_memories: list[ScoredMemory]) -> MemoryAction: ...
```

### 2.2 LLMClient (§5.1 VERBATIM — do NOT modify llm.py)

```python
# ---------- src/brain/llm.py ----------
class LLMClient(ABC):
    @abstractmethod
    async def chat_json(self, messages: list[dict], schema: dict,
                        temperature: float = 0.0) -> dict: ...
    # messages: OpenAI-style [{"role", "content"}]
    # schema: JSON-Schema dict
    # returns: the parsed dict
```

Call signature used by `LLMReconciler`:

```python
result = await self._llm.chat_json(messages, schema=DECISION_SCHEMA, temperature=0.0)
```

### 2.3 MemoryStore (§5.1 VERBATIM — use existing methods, add NO new ones)

```python
# ---------- src/brain/store/base.py ----------
class MemoryStore(ABC):
    async def add(self, content: str, scope: Scope, metadata: dict | None = None) -> Memory: ...
    async def search(self, query: str, scope: Scope, limit: int = 10) -> list[ScoredMemory]: ...
    async def get(self, id: str, scope: Scope) -> Memory | None: ...
    async def delete(self, id: str, scope: Scope) -> bool: ...
    async def update(self, id: str, content: str, scope: Scope) -> Memory | None: ...
```

### 2.4 MemoryService facade (§5.1 VERBATIM — L2 shipped `add`; L3 adds search/get/forget + build_memory)

```python
# ---------- src/brain/memory.py ----------
class MemoryService:
    def __init__(self, store: MemoryStore, llm: LLMClient,
                 reconciler: Reconciler, *, search_k: int = 5): ...
    async def add(self, messages: list[dict], scope: Scope) -> list[Memory]: ...  # extract → reconcile → dispatch (write-loop; DO NOT EDIT)
    async def search(self, query: str, scope: Scope, limit: int = 10) -> list[ScoredMemory]: ...
    async def get(self, id: str, scope: Scope) -> Memory | None: ...
    async def forget(self, id: str, scope: Scope) -> bool: ...

def build_memory() -> MemoryService: ...   # wires store+embedder+llm+LLMReconciler from settings
```

### 2.5 LLMReconciler (new in L3)

```python
# ---------- src/brain/reconcile.py (add alongside AlwaysAddReconciler) ----------
class LLMReconciler(Reconciler):
    def __init__(self, llm: LLMClient): ...

    async def reconcile(
        self,
        candidate: FactCandidate,
        similar_memories: list[ScoredMemory],
    ) -> MemoryAction: ...
```

- `LLMReconciler` has NO embedder, NO store, NO candidate-fetch. The caller (the write-loop, already in L2) passes `similar_memories` pre-fetched via `store.search`.
- The only external call is `self._llm.chat_json(...)`.
- `target_id` resolution: LLM returns `target_index` (0-based int into `similar_memories`); map as `similar_memories[target_index].memory.id`. If `target_index` is `null`, out of range (< 0 or >= len), or the list is empty, fall back to `MemoryActionKind.ADD` with `target_id=None`.
- `target_id` is always a `str` (it comes from `Memory.id` which is `str(uuid.uuid4())`).

### 2.6 Reconciliation prompt

```
System:
You are a memory reconciliation engine. Given a NEW FACT and a list of EXISTING MEMORIES,
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

User:
NEW FACT: {candidate.content}

EXISTING MEMORIES:
{numbered list: "0. {sm.memory.content}" for each sm in similar_memories, or "(none)" if empty}
```

Render the numbered list as one item per line, 0-indexed, e.g.:

```
0. User likes pizza
1. User enjoys Italian food
```

If `similar_memories` is empty, render `(none)` and the LLM must return `action: "ADD"`.

### 2.7 Decision JSON schema

```python
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action":       {"type": "string", "enum": ["ADD", "UPDATE", "DELETE", "NOOP"]},
        "target_index": {"type": ["integer", "null"]},
        "content":      {"type": ["string", "null"]},
        "reason":       {"type": ["string", "null"]},
    },
    "required": ["action"],
}
```

The `reason` field is for debugging; it is never stored.

---

## 3. Implementation order

Each numbered step leaves the codebase in a working, runnable state.

**Step 1 — `LLMReconciler` skeleton (reconcile.py)**

Add `LLMReconciler` to `src/brain/reconcile.py` alongside the existing `AlwaysAddReconciler`. Implement:
- `__init__(self, llm: LLMClient)` — store `self._llm = llm`.
- `reconcile(candidate, similar_memories)` — build prompt, call `self._llm.chat_json(messages, DECISION_SCHEMA, temperature=0.0)`, parse response, resolve `target_index` → `target_id` (str), return `MemoryAction`.
- Define `DECISION_SCHEMA` as a module-level constant.
- `LLMReconciler` has no other instance variables.

At this point: `LLMReconciler` is importable and unit-testable; nothing in `MemoryService` is changed yet.

**Step 2 — Unit tests for decision logic (test_reconcile.py)**

Write `tests/test_reconcile.py` using `StubLLM` (a local `LLMClient` subclass that returns a hard-coded dict without calling Ollama):

- `test_update_decision` — inject `similar_memories=[pizza_scored_memory]`, `StubLLM` returns `{"action":"UPDATE","target_index":0,"content":"User prefers sushi","reason":"..."}`. Assert `action.kind == MemoryActionKind.UPDATE`, `action.target_id == pizza_memory.id`, `action.content == "User prefers sushi"`.
- `test_delete_decision` — inject one `similar_memories`, `StubLLM` returns `{"action":"DELETE","target_index":0,"content":null,"reason":"..."}`. Assert `action.kind == MemoryActionKind.DELETE`, `action.target_id == pizza_memory.id`, `action.content is None`.
- `test_noop_decision` — `StubLLM` returns `{"action":"NOOP","target_index":0,"content":null,"reason":"..."}`. Assert `action.kind == MemoryActionKind.NOOP`.
- `test_add_decision_empty_candidates` — `similar_memories=[]`, `StubLLM` returns `{"action":"ADD","target_index":null,"content":"User likes tacos","reason":"..."}`. Assert `action.kind == MemoryActionKind.ADD`, `action.target_id is None`.
- `test_out_of_range_target_index_falls_back_to_add` — `StubLLM` returns `{"action":"UPDATE","target_index":99,"content":"..."}` but `similar_memories` has only 1 entry. Assert fallback: `action.kind == MemoryActionKind.ADD`, `action.target_id is None`.

All five tests pass with no Ollama, no embedder, no DB.

**Step 3 — Wire LLMReconciler into MemoryService + add pass-throughs (memory.py)**

Edit `src/brain/memory.py`:
- Add `search`, `get`, `forget` pass-through methods that delegate directly to `self._store` (no logic).
- Add `build_memory()` factory: constructs `SQLiteMemoryStore` (with the local embedder), `OllamaLLMClient`, and `LLMReconciler(llm)`, then returns `MemoryService(store, llm, LLMReconciler(llm), search_k=5)`.
- The write-loop (`add`) is NOT edited. Only the `reconciler` argument passed at construction changes (from `AlwaysAddReconciler()` to `LLMReconciler(llm)` in `build_memory()`). Existing tests that construct `MemoryService` with `AlwaysAddReconciler` directly continue to pass.

**Step 4 — Recorded action fixture + deterministic integration test**

Create `tests/fixtures/llm_responses/reconcile_pizza_sushi.json`:

```json
[
  {
    "call": "extract",
    "response": {"facts": ["User prefers sushi"]}
  },
  {
    "call": "reconcile",
    "response": {
      "action": "UPDATE",
      "target_index": 0,
      "content": "User prefers sushi",
      "reason": "User changed food preference from pizza to sushi"
    }
  }
]
```

Write `test_pizza_sushi_update_integration` in `tests/test_reconcile.py`:
- Use the real sqlite-vec temp DB fixture (from `conftest.py`, same as L1/L2 tests), fake embedder, and a `StubLLM` that replays `reconcile_pizza_sushi.json`.
- Steps:
  1. Directly call `store.add("User likes pizza", scope)` → record `pizza_id` from the returned `Memory.id`.
  2. Construct `MemoryService(store, stub_llm, LLMReconciler(stub_llm), search_k=5)`.
  3. Call `service.add([{"role":"user","content":"Actually I prefer sushi now"}], scope)`.
  4. **Assert action sequence:** the reconciler was called once; the action returned was `kind=UPDATE`, `target_id == pizza_id`, `content == "User prefers sushi"`.
  5. **Assert final store state:** `store.search("food preference", scope, limit=10)` returns exactly one result; `result[0].memory.content == "User prefers sushi"`; no memory with content containing `"pizza"` exists (updated in place, not duplicated).

The `StubLLM` for this test must intercept both the extraction call (return one fact `"User prefers sushi"`) and the reconciliation call (return the fixture JSON above).

**Step 5 — Live smoke test**

Write `test_layer3_live_smoke` in `tests/test_reconcile.py`, marked `@pytest.mark.live` (skipped by default; enabled with `-m live`):
- Use real sqlite-vec temp DB, real local embedder, real `OllamaLLMClient(model="qwen2.5:3b-instruct")`.
- Seed "User likes pizza" via `store.add`.
- Call `service.add([{"role":"user","content":"Actually I prefer sushi now"}], scope)`.
- **Assert (structural only):** `len(returned_memories) >= 1`; each returned memory has `id` (non-empty str), `content` (non-empty str), `user_id == scope.user_id`.
- Do NOT assert action kind or exact content (LLM nondeterminism).

**Step 6 — Demo script**

Write `scripts/demo_layer3.py`:
- Calls `build_memory()` to get a `MemoryService`.
- Seeds "User likes pizza" directly via the store (access via `service._store` or expose store as a param; direct `store.add` avoids LLM extraction for the seed step).
- Calls `service.add([{"role":"user","content":"Actually I prefer sushi now"}], scope)`.
- Calls `service.search("food preference", scope)` and prints results.
- Asserts (printed + raises if wrong): exactly one memory returned; content contains "sushi"; no memory content contains "pizza".

---

## 4. Test fixtures

### Fixture paths

```
tests/fixtures/
  conversations/
    pizza_to_sushi.json          # input messages for integration test
  llm_responses/
    reconcile_pizza_sushi.json   # recorded extraction + reconciliation responses (the oracle)
```

### `pizza_to_sushi.json` (input conversation)

```json
[
  {"role": "user", "content": "Actually I prefer sushi now"}
]
```

### `reconcile_pizza_sushi.json` (oracle — the StubLLM replay fixture)

```json
[
  {
    "call": "extract",
    "response": {
      "facts": ["User prefers sushi"]
    }
  },
  {
    "call": "reconcile",
    "response": {
      "action": "UPDATE",
      "target_index": 0,
      "content": "User prefers sushi",
      "reason": "User changed food preference from pizza to sushi"
    }
  }
]
```

The `StubLLM` in the integration test replays these in order: first call returns the extract response, second call returns the reconcile response.

### StubLLM (defined inline in test file or conftest.py)

```python
class StubLLM(LLMClient):
    def __init__(self, responses: list[dict]):
        self._responses = iter(responses)

    async def chat_json(self, messages: list[dict], schema: dict,
                        temperature: float = 0.0) -> dict:
        return next(self._responses)
```

No Ollama, no network, no embedder.

---

## 5. Pinned values and assertions

| Parameter | Value | Authority |
|---|---|---|
| LLM model | `qwen2.5:3b-instruct` | Same `LLM_MODEL` as L2 (CP-3); from `settings.llm_model`; passed to `OllamaLLMClient` |
| Cosine threshold for candidates | **0.3** | CP-4; filter applied in the write-loop before passing `similar_memories` to `reconcile` |
| k (candidate count) | **5** | CP-4; `MemoryService.__init__(search_k=5)` |
| LLM temperature | **0.0** | `chat_json(..., temperature=0.0)` |
| `target_id` type | **`str`** | Always `similar_memories[i].memory.id`; `Memory.id` is `str(uuid.uuid4())` |

### Asserted action sequence (integration test oracle)

Given:
- Store contains one memory: `"User likes pizza"` (id = `pizza_id`)
- `similar_memories` injected = `[ScoredMemory(memory=pizza_memory, score=0.85)]` (score above threshold 0.3)
- `StubLLM` returns `{"action":"UPDATE","target_index":0,"content":"User prefers sushi","reason":"..."}`

Expected `MemoryAction` returned by `reconcile`:
```python
MemoryAction(
    kind=MemoryActionKind.UPDATE,
    content="User prefers sushi",
    target_id=pizza_id,   # str, == pizza_memory.id
    metadata={},
)
```

### Asserted final store state (integration test)

After `service.add([{"role":"user","content":"Actually I prefer sushi now"}], scope)`:

```python
results = await store.search("food preference", scope, limit=10)
assert len(results) == 1
assert results[0].memory.content == "User prefers sushi"
# No row with content "User likes pizza" survives:
assert all("pizza" not in r.memory.content.lower() for r in results)
```

The pizza memory must be UPDATED IN PLACE (same `id = pizza_id`, `content` changed, `updated_at` bumped) — no contradictory duplicate row.

---

## 6. Layer completion verification

**Done when:**

```
uv run python scripts/demo_layer3.py
```

Expected output (exact strings may vary; these facts must hold):
```
Seeded: "User likes pizza" (id=<uuid>)
Fed: "Actually I prefer sushi now"
Action: UPDATE  target_id=<same-uuid>  content="User prefers sushi"
Final memories (1):
  [0] "User prefers sushi"  score=...
PASS: pizza updated to sushi in place, no duplicate.
```

And:

```
uv run pytest tests/test_reconcile.py -v -m "not live"
```

Expected: all 6 deterministic tests pass (5 unit + 1 integration); 0 failures; 0 network calls.

**Hands off to Layer 4:** the complete async `MemoryService` facade — `add(messages, scope)`, `search(query, scope, limit)`, `get(id, scope)`, `forget(id, scope)` — plus `build_memory() -> MemoryService`, all taking `Scope` objects, all async, `id` always `str`. Layer 4 constructs via `build_memory()` and calls only these four methods.
