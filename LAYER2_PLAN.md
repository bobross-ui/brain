# Brain — Layer 2: Extraction + naive add — Implementation Spec

> Autonomous build spec. Self-contained — implement exactly what's here; depend on no other file.
> §5.1 of IMPLEMENTATION_PLAN.md is authoritative. Every type restated here is verbatim from §5.1.
> Do NOT read LAYER1_PLAN.md or any other layer plan.

---

## 0. Scope

**Build (in scope):**
- `src/brain/llm.py` — `LLMClient` ABC + `OllamaLLMClient` + `FakeLLMClient`
- `src/brain/models.py` additions — `FactCandidate`, `MemoryActionKind`, `MemoryAction`, `Reconciler` ABC (append to whatever L1 left; do NOT redefine `Scope`, `Memory`, `ScoredMemory`)
- `src/brain/extract.py` — `Extractor`: conversation messages → `list[FactCandidate]`
- `src/brain/reconcile.py` — `AlwaysAddReconciler` (implements `Reconciler`; always returns `MemoryAction(kind=MemoryActionKind.ADD, ...)`)
- `src/brain/memory.py` — `MemoryService` facade + `build_memory()` factory; write-loop with ALL FOUR `action.kind` branches (ADD / UPDATE / DELETE / NOOP)
- `scripts/demo_layer2.py` — fixture-driven demo + live smoke
- `tests/test_extract.py` — deterministic oracle test (recorded fixture) + live smoke
- `tests/fixtures/conversations/conv_01.json` — fixture conversation
- `tests/fixtures/llm_responses/extract_conv_01.json` — recorded extraction response (the oracle)

**Do NOT build (out of scope):**
- Deduplication or content-hash rejection (no UNIQUE check — duplicates are allowed; L3 owns policy)
- LLM-backed reconciliation logic (L3)
- Similarity threshold tuning for reconciliation (CP-4, L3)
- MCP server (L4)
- Any change to L1's `MemoryStore`, `SQLiteMemoryStore`, `Embedder`, or SQLite schema
- HTTP transport, auth, or multi-tenancy beyond scoped store calls
- Advanced retrieval (hybrid, BM25, rerank)

---

## 1. Setup — do this FIRST

### 1a. Verify Ollama is installed and the model is pulled

```bash
# Install Ollama if not present: https://ollama.com/download
ollama --version                              # must succeed
ollama pull qwen2.5:3b-instruct               # pull once; subsequent calls use local cache
ollama run qwen2.5:3b-instruct "Reply only: ok"   # confirm model responds
```

### 1b. Add the `ollama` package to the project

```bash
uv add "ollama>=0.6.2"
```

Confirm `pyproject.toml` now lists `ollama>=0.6.2` under `[project.dependencies]`.

### 1c. Add `LLM_MODEL` and `LLM_PROVIDER` to config and `.env.example`

In `src/brain/config.py`, add two settings fields (append; do not change existing fields):

```python
llm_model: str = "qwen2.5:3b-instruct"   # shared by extraction (L2) AND reconciliation (L3)
llm_provider: str = "ollama"              # default; the only supported value in v0
```

In `.env.example`, append:

```
LLM_MODEL=qwen2.5:3b-instruct
LLM_PROVIDER=ollama
```

### 1d. Files touched / created this layer

| File | Action |
|---|---|
| `pyproject.toml` | append `ollama>=0.6.2` dep |
| `.env.example` | append `LLM_MODEL`, `LLM_PROVIDER` |
| `src/brain/config.py` | append two fields |
| `src/brain/models.py` | append `FactCandidate`, `MemoryActionKind`, `MemoryAction`, `Reconciler` |
| `src/brain/llm.py` | create (new file) |
| `src/brain/extract.py` | create (new file) |
| `src/brain/reconcile.py` | create (new file) |
| `src/brain/memory.py` | create (new file) |
| `scripts/demo_layer2.py` | create (new file) |
| `tests/test_extract.py` | create (new file) |
| `tests/fixtures/conversations/conv_01.json` | create (new file) |
| `tests/fixtures/llm_responses/extract_conv_01.json` | create (new file) |

---

## 2. Interfaces & data contracts

Everything in this section is **verbatim from §5.1**. Conform all call sites to these exact signatures. Change nothing.

### 2.1 Types already defined in `models.py` by L1 (consume, do NOT redefine)

```python
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
```

### 2.2 Types L2 adds to `models.py` (append verbatim)

```python
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
    target_id: str | None = None        # a Memory.id; set for UPDATE / DELETE
    metadata: dict = Field(default_factory=dict)

class Reconciler(ABC):                  # the reconciliation seam (lives in models.py)
    @abstractmethod
    async def reconcile(self, candidate: FactCandidate,
                        similar_memories: list[ScoredMemory]) -> MemoryAction: ...
```

Required imports to prepend to `models.py` (add only what is missing):

```python
from abc import ABC, abstractmethod
from enum import Enum
```

### 2.3 `LLMClient` interface and implementations — `src/brain/llm.py`

```python
# src/brain/llm.py
from abc import ABC, abstractmethod
import ollama                          # ONLY this file imports ollama

class LLMClient(ABC):
    @abstractmethod
    async def chat_json(self, messages: list[dict], schema: dict,
                        temperature: float = 0.0) -> dict: ...
    # messages: OpenAI-style [{"role": "...", "content": "..."}]
    # schema: JSON-Schema dict passed as format= to Ollama
    # returns: the parsed dict (call json.loads on Ollama's response.message.content)

class OllamaLLMClient(LLMClient):
    def __init__(self, model: str): ...   # stores self._model = model

    async def chat_json(self, messages: list[dict], schema: dict,
                        temperature: float = 0.0) -> dict:
        # client = ollama.AsyncClient()
        # response = await client.chat(
        #     model=self._model,
        #     messages=messages,
        #     format=schema,
        #     options={"temperature": temperature},
        # )
        # return json.loads(response.message.content)
        ...

class FakeLLMClient(LLMClient):
    """Returns a pre-recorded dict. Used in deterministic tests."""
    def __init__(self, recorded: dict): ...   # stores self._recorded = recorded

    async def chat_json(self, messages: list[dict], schema: dict,
                        temperature: float = 0.0) -> dict:
        return self._recorded
```

### 2.4 Extraction prompt and output JSON schema

#### Extraction system prompt (literal — characters carry meaning)

```
You are a memory extraction assistant. Given a conversation, extract all distinct atomic facts about the user. Each fact must be a single, self-contained statement. Output only JSON matching the provided schema.
```

#### Extraction user prompt template (literal)

```
Extract all atomic facts about the user from the following conversation.

Conversation:
{conversation_text}
```

Where `{conversation_text}` is built by joining each message as `"{role}: {content}"`, one per line.

#### Extraction output JSON schema (literal — exact characters)

```json
{
  "type": "object",
  "properties": {
    "facts": {
      "type": "array",
      "items": {"type": "string"}
    }
  },
  "required": ["facts"],
  "additionalProperties": false
}
```

Parsing: `response_dict["facts"]` → `list[str]` → for each fact string `f`:
`FactCandidate(content=f, metadata={"source": "extraction"})`.

### 2.5 `Extractor` — `src/brain/extract.py`

```python
# src/brain/extract.py
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
    def __init__(self, llm: LLMClient): ...

    async def extract(self, messages: list[dict]) -> list[FactCandidate]:
        # 1. conversation_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        # 2. llm_messages = [
        #        {"role": "system", "content": SYSTEM_PROMPT},
        #        {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(conversation_text=conversation_text)},
        #    ]
        # 3. response = await self._llm.chat_json(llm_messages, schema=EXTRACTION_SCHEMA, temperature=0.0)
        # 4. return [FactCandidate(content=f, metadata={"source": "extraction"})
        #            for f in response["facts"]]
        ...
```

### 2.6 `AlwaysAddReconciler` — `src/brain/reconcile.py`

```python
# src/brain/reconcile.py
from brain.models import FactCandidate, MemoryAction, MemoryActionKind, Reconciler, ScoredMemory

class AlwaysAddReconciler(Reconciler):
    async def reconcile(self, candidate: FactCandidate,
                        similar_memories: list[ScoredMemory]) -> MemoryAction:
        return MemoryAction(
            kind=MemoryActionKind.ADD,
            content=candidate.content,
            metadata=candidate.metadata,
        )
```

No other logic. `similar_memories` is accepted (correct param name) but ignored.

### 2.7 `MemoryService` facade — `src/brain/memory.py`

```python
# src/brain/memory.py
from brain.extract import Extractor
from brain.llm import LLMClient, OllamaLLMClient
from brain.models import Memory, MemoryAction, MemoryActionKind, Reconciler, Scope, ScoredMemory
from brain.reconcile import AlwaysAddReconciler
from brain.store.base import MemoryStore

class MemoryService:
    def __init__(self, store: MemoryStore, llm: LLMClient,
                 reconciler: Reconciler, *, search_k: int = 5):
        self._store = store
        self._llm = llm
        self._reconciler = reconciler
        self._search_k = search_k
        self._extractor = Extractor(llm)   # owned internally; callers never construct Extractor

    async def add(self, messages: list[dict], scope: Scope) -> list[Memory]:
        candidates = await self._extractor.extract(messages)
        stored: list[Memory] = []
        for candidate in candidates:
            similar = await self._store.search(candidate.content, scope, limit=self._search_k)
            action: MemoryAction = await self._reconciler.reconcile(candidate, similar)
            if action.kind == MemoryActionKind.ADD:
                memory = await self._store.add(action.content, scope, action.metadata)
                stored.append(memory)
            elif action.kind == MemoryActionKind.UPDATE:
                await self._store.update(action.target_id, action.content, scope)
            elif action.kind == MemoryActionKind.DELETE:
                await self._store.delete(action.target_id, scope)
            elif action.kind == MemoryActionKind.NOOP:
                pass
        return stored

    async def search(self, query: str, scope: Scope, limit: int = 10) -> list[ScoredMemory]:
        return await self._store.search(query, scope, limit)

    async def get(self, id: str, scope: Scope) -> Memory | None:
        return await self._store.get(id, scope)

    async def forget(self, id: str, scope: Scope) -> bool:
        return await self._store.delete(id, scope)

def build_memory() -> MemoryService:
    # Wire from settings:
    #   from brain.config import settings
    #   from brain.embeddings import LocalEmbedder
    #   from brain.store.sqlite import SQLiteMemoryStore
    #   embedder   = LocalEmbedder(settings.embedder_model)
    #   store      = SQLiteMemoryStore(settings.db_path, embedder)
    #   llm        = OllamaLLMClient(settings.llm_model)
    #   reconciler = AlwaysAddReconciler()
    #   return MemoryService(store, llm, reconciler)
    ...
```

**Important:** Pass `Scope` objects to all store calls — never dicts or loose kwargs.

**Important:** `id` / `target_id` is always `str`. Never pass a `uuid.UUID` object.

**Important:** ALL FOUR `action.kind` branches (ADD, UPDATE, DELETE, NOOP) must be present in the write-loop — L3 will NOT edit this loop; it only injects a new `Reconciler` implementation.

---

## 3. Implementation order

Each step leaves the system in a working, runnable state.

**Step 1 — Install dep + configure**
- `uv add "ollama>=0.6.2"`
- Append `llm_model` and `llm_provider` to `config.py` and `.env.example`.
- Verify: `python -c "import ollama; print('ok')"` succeeds.

**Step 2 — Extend `models.py`**
- Append `FactCandidate`, `MemoryActionKind`, `MemoryAction`, `Reconciler` (verbatim from §2.2).
- Add missing imports (`ABC`, `abstractmethod`, `Enum`) if not already present.
- Verify: `python -c "from brain.models import FactCandidate, MemoryActionKind, MemoryAction, Reconciler; print('ok')"`.

**Step 3 — Create `llm.py`**
- Define `LLMClient` ABC, `OllamaLLMClient`, `FakeLLMClient` (§2.3).
- Verify: `python -c "from brain.llm import LLMClient, OllamaLLMClient, FakeLLMClient; print('ok')"`.

**Step 4 — Create `extract.py`**
- Define `Extractor` with `SYSTEM_PROMPT`, `USER_PROMPT_TEMPLATE`, `EXTRACTION_SCHEMA` as module-level constants (§2.4, §2.5).
- Verify: `python -c "from brain.extract import Extractor; print('ok')"`.

**Step 5 — Create test fixtures**
- Write `tests/fixtures/conversations/conv_01.json` (§4.1).
- Write `tests/fixtures/llm_responses/extract_conv_01.json` (§4.2).
- Verify: both files parse as valid JSON.

**Step 6 — Create `tests/test_extract.py`**
- Deterministic oracle test using `FakeLLMClient` + recorded fixture (§5).
- Live smoke test using `OllamaLLMClient` marked `@pytest.mark.ollama` (run separately).
- Run deterministic tests: `uv run pytest tests/test_extract.py -k "not ollama" -v` → PASSED.

**Step 7 — Create `reconcile.py`**
- Define `AlwaysAddReconciler` (§2.6).
- Verify: `python -c "from brain.reconcile import AlwaysAddReconciler; print('ok')"`.

**Step 8 — Create `memory.py`**
- Define `MemoryService` + `build_memory()` (§2.7).
- ALL FOUR dispatch branches present (ADD, UPDATE, DELETE, NOOP).
- Verify: `python -c "from brain.memory import MemoryService, build_memory; print('ok')"`.

**Step 9 — Create `scripts/demo_layer2.py`**
- Fixture-driven path (`--mode fixture`): uses `FakeLLMClient` + recorded response + real sqlite-vec DB → assert exact fact set.
- Live smoke path (`--mode live`): uses `build_memory()` (real Ollama) → asserts ≥1 fact stored.
- Run fixture path: `uv run python scripts/demo_layer2.py --mode fixture` → prints stored facts, exits 0.

**Step 10 — Final verification**
- Run the full done-when command sequence (§6).

---

## 4. Test fixtures

### 4.1 Fixture conversation — `tests/fixtures/conversations/conv_01.json`

```json
[
  {"role": "user",      "content": "Hey, I just moved to Berlin last month."},
  {"role": "assistant", "content": "Nice! How are you finding it?"},
  {"role": "user",      "content": "I love it. I work as a software engineer and I cycle to work every day."},
  {"role": "assistant", "content": "That sounds great. Any hobbies outside work?"},
  {"role": "user",      "content": "I play chess in the evenings and I am vegetarian."}
]
```

### 4.2 Recorded extraction response — `tests/fixtures/llm_responses/extract_conv_01.json`

This is the **oracle** — the exact dict `FakeLLMClient` returns. Matches the extraction schema exactly.

```json
{
  "facts": [
    "The user moved to Berlin last month.",
    "The user works as a software engineer.",
    "The user cycles to work every day.",
    "The user plays chess in the evenings.",
    "The user is vegetarian."
  ]
}
```

**Pinned fact count:** 5 facts. These five strings are the oracle — tests assert against them exactly (set membership, not order).

---

## 5. Pinned values and assertions

| Item | Pinned value |
|---|---|
| Ollama Python package | `ollama>=0.6.2` |
| LLM model | `qwen2.5:3b-instruct` (exact string; settings key `llm_model`) |
| `LLMClient` method name | `chat_json` (NOT `complete_json` or any other name) |
| Facade class name | `MemoryService` (NOT `Memory`) |
| `Reconciler` param name | `similar_memories` (not `similar`, not `candidates`) |
| `MemoryAction` dispatch field | `action.kind` (a `MemoryActionKind` enum member) |
| `store.search` call site | `store.search(candidate.content, scope, limit=self._search_k)` — `scope` is a `Scope` object |
| `store.add` call site | `store.add(action.content, scope, action.metadata)` |
| `store.update` call site | `store.update(action.target_id, action.content, scope)` |
| `store.delete` call site | `store.delete(action.target_id, scope)` |
| `id` / `target_id` type | `str` — never `uuid.UUID` |
| `FactCandidate.metadata` for extracted facts | `{"source": "extraction"}` |
| Fixture fact count | **5** |
| Fixture facts (exact, set membership) | See §4.2 |
| Live smoke assertion | `len(stored) >= 1` and `stored[0].content` is a non-empty string |
| `content_hash` policy | stored on every row; NOT a UNIQUE constraint; duplicates allowed |
| Dispatch branches in write-loop | ADD, UPDATE, DELETE, NOOP — all four present in `MemoryService.add` |
| `ollama` import location | ONLY `src/brain/llm.py` — no other module imports `ollama` |

### Deterministic oracle test assertions (`tests/test_extract.py`)

```python
# Test 1: Extractor unit test
# Given FakeLLMClient(recorded=RECORDED_RESPONSE) and conv_01 messages:
candidates = await extractor.extract(messages)
assert len(candidates) == 5
contents = {c.content for c in candidates}
assert "The user moved to Berlin last month." in contents
assert "The user works as a software engineer." in contents
assert "The user cycles to work every day." in contents
assert "The user plays chess in the evenings." in contents
assert "The user is vegetarian." in contents
for c in candidates:
    assert c.metadata == {"source": "extraction"}

# Test 2: MemoryService.add integration test
# Given FakeLLMClient + FakeEmbedder (L1) + real sqlite-vec temp DB + AlwaysAddReconciler:
scope = Scope(user_id="test-user")
stored = await service.add(messages, scope)
assert len(stored) == 5
stored_contents = {m.content for m in stored}
assert "The user moved to Berlin last month." in stored_contents
assert "The user works as a software engineer." in stored_contents
assert "The user cycles to work every day." in stored_contents
assert "The user plays chess in the evenings." in stored_contents
assert "The user is vegetarian." in stored_contents
for m in stored:
    assert isinstance(m.id, str)
    assert m.user_id == "test-user"
    assert m.namespace == "default"
```

---

## 6. Layer completion verification

**Done when:**

Step A — deterministic oracle (no Ollama needed):

```bash
uv run pytest tests/test_extract.py -k "not ollama" -v
```

Expected:

```
PASSED tests/test_extract.py::test_extract_oracle
PASSED tests/test_extract.py::test_memory_service_add_oracle
2 passed in <N>s
```

Step B — live smoke (requires `ollama serve` running with `qwen2.5:3b-instruct` pulled):

```bash
uv run pytest tests/test_extract.py -k "ollama" -v -s
```

Expected: `test_extract_live_smoke` and/or `test_memory_service_add_live_smoke` pass; output confirms `stored >= 1 fact`.

Step C — fixture demo:

```bash
uv run python scripts/demo_layer2.py --mode fixture
```

Expected: prints 5 stored memories with the exact fact strings from §4.2, exits 0.

**Hands off to Layer 3:**
- `FactCandidate` (defined in `models.py`, produced by `Extractor` in `extract.py`)
- `Reconciler` ABC (defined in `models.py`, param name `similar_memories`)
- `MemoryActionKind` / `MemoryAction` (defined in `models.py`)
- `AlwaysAddReconciler` in `reconcile.py` (L3 adds `LLMReconciler` alongside it; does NOT edit `AlwaysAddReconciler`)
- `MemoryService` write-loop in `memory.py` with all four `action.kind` branches — **L3 does NOT edit this loop**; it only injects a new `Reconciler` implementation
- `LLMClient` / `OllamaLLMClient` / `FakeLLMClient` in `llm.py` — L3 reuses the same `chat_json` method for reconciliation decisions
- `build_memory()` factory in `memory.py` — L4 calls this as its construction entry point
