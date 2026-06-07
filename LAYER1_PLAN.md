# Brain — Layer 1: Storage + Retrieval foundation — Implementation Spec

> Autonomous build spec. Self-contained — implement exactly what's here; depend on no other file.
> §5.1 of IMPLEMENTATION_PLAN.md is the authoritative type contract; every type/signature below is
> restated verbatim from it. If anything in this file conflicts with §5.1, §5.1 wins.

---

## 0. Scope

**Build (in scope):**
- `pyproject.toml` (uv-managed) with all L1 deps pinned
- `src/brain/config.py` — pydantic-settings `Settings` object (DB path, embedder model name)
- `src/brain/models.py` — `Scope`, `Memory`, `ScoredMemory`, `FactCandidate`, `MemoryActionKind`, `MemoryAction`, `Reconciler` (restated verbatim from §5.1; the L2/L3 types are stubs here so every layer can import from one module)
- `src/brain/embeddings.py` — `Embedder` ABC + `SentenceTransformerEmbedder` (local/free) + `FakeEmbedder` (deterministic, for tests)
- `src/brain/store/schema.sql` — DDL for `memories` companion table + `vec_memories` vec0 virtual table
- `src/brain/store/base.py` — `MemoryStore` ABC (5 methods, verbatim from §5.1)
- `src/brain/store/sqlite.py` — `SQLiteMemoryStore` implementing all 5 methods; connection-per-operation; asyncio.to_thread wrapping
- `scripts/demo_layer1.py` — runnable demo: inserts 5 alice facts + 2 bob facts, queries as alice, prints cosine-ranked results
- `tests/conftest.py` — tmp_path DB fixture, FakeEmbedder fixture
- `tests/test_store.py` — unit tests: similarity ordering, scope isolation, CRUD, update

**Do NOT build (out of scope — name explicitly to avoid drift):**
- LLM fact extraction (`extract.py`) — Layer 2
- `LLMClient` / `llm.py` implementation — Layer 2 (CP-3)
- `AlwaysAddReconciler` / `LLMReconciler` — Layer 2/3
- Reconciliation / dedup logic — Layer 3 (`content_hash` is stored but NOT used to reject or merge)
- `MemoryService` facade (`memory.py`) — Layer 2/3 hand-off
- `build_memory()` factory — Layer 3/4 hand-off
- MCP server (`mcp_server.py`) — Layer 4
- `demo_layer2/3/4.py` scripts
- HTTP/SSE transport, auth
- Advanced retrieval (hybrid search, reranking, recency decay)

---

## 1. Setup — do this FIRST

### 1.1 Python + package manager

```
Python 3.11+
uv (https://docs.astral.sh/uv/)
```

### 1.2 `pyproject.toml` — pinned deps for L1

```toml
[project]
name = "brain"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "sqlite-vec==0.1.9",
    "sentence-transformers>=2.7,<3.0",
    "pydantic>=2.7,<3.0",
    "pydantic-settings>=2.3,<3.0",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

Run `uv sync --extra dev` to install.

### 1.3 sqlite-vec — pinned version and load strategy

**Pinned version: `sqlite-vec==0.1.9`** (stable release; do not use a pre-release).

Load sequence — executed once per connection, on the same thread that opened it:

```python
import sqlite3
import sqlite_vec

def _open_db(path: str) -> sqlite3.Connection:
    db = sqlite3.connect(path, isolation_level=None)   # autocommit
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)   # lock back down
    db.row_factory = sqlite3.Row
    return db
```

### 1.4 Connection / threading strategy — CP-1c

**Connection-per-operation.** Every store method opens a connection at the start of its
`asyncio.to_thread` callable and closes it (via `with` / explicit `.close()`) before returning.
The sqlite-vec extension is loaded per connection. `check_same_thread` is left at its default
(`True`) — safe because the connection never leaves the worker thread. No shared connection
object, no pool, no single dedicated worker thread.

Rationale: avoids "SQLite objects created in a thread can only be used in that same thread"
errors across the asyncio thread pool; extension loaded per connection guarantees it is always
present; the overhead is negligible for v0 throughput.

### 1.5 `.env.example`

```
BRAIN_DB_PATH=./brain.db
BRAIN_EMBEDDER_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 1.6 Directory bootstrap

```
brain/
  pyproject.toml
  uv.lock
  .env.example
  src/brain/
    __init__.py
    config.py
    models.py
    embeddings.py
    store/
      __init__.py
      base.py
      schema.sql
      sqlite.py
  scripts/
    demo_layer1.py
  tests/
    conftest.py
    test_store.py
```

---

## 2. Interfaces & data contracts

### 2.1 §5.1 Pinned types — restated VERBATIM

```python
# ---------- identifiers ----------
# Every `id` / `target_id` is a STRING: str(uuid.uuid4()), stored in a TEXT column.
# A uuid.UUID object must NEVER cross any store / facade / tool boundary — convert to str first.

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
    target_id: str | None = None        # a Memory.id; set for UPDATE / DELETE
    metadata: dict = Field(default_factory=dict)

class Reconciler(ABC):                  # the reconciliation seam (lives in models.py)
    @abstractmethod
    async def reconcile(self, candidate: FactCandidate,
                        similar_memories: list[ScoredMemory]) -> MemoryAction: ...

# ---------- src/brain/embeddings.py ----------
class Embedder(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...   # D-dim vector; D pinned at CP-2

# ---------- src/brain/llm.py ----------
class LLMClient(ABC):                   # ONE method, used by BOTH extraction (L2) and reconciliation (L3)
    @abstractmethod
    async def chat_json(self, messages: list[dict], schema: dict,
                        temperature: float = 0.0) -> dict: ...
    # messages: OpenAI-style [{"role","content"}]; schema: JSON-Schema dict; returns the parsed dict.
    # Default impl: OllamaLLMClient(model=settings.llm_model). Provider/model = CP-3 (L2 owns); L3 REUSES it.

# ---------- src/brain/store/base.py ----------
class MemoryStore(ABC):                 # storage swap seam; SQLiteMemoryStore implements it; holds an Embedder, embeds internally
    @abstractmethod
    async def add(self, content: str, scope: Scope, metadata: dict | None = None) -> Memory: ...
    @abstractmethod
    async def search(self, query: str, scope: Scope, limit: int = 10) -> list[ScoredMemory]: ...
    @abstractmethod
    async def get(self, id: str, scope: Scope) -> Memory | None: ...
    @abstractmethod
    async def delete(self, id: str, scope: Scope) -> bool: ...
    @abstractmethod
    async def update(self, id: str, content: str, scope: Scope) -> Memory | None: ...
    # All five implemented & unit-tested in L1. `update` is unused by any write path until L3 wires it in.

# ---------- src/brain/memory.py ----------
class MemoryService:                    # the product FACADE — named MemoryService, NOT `Memory` (that's the record)
    def __init__(self, store: MemoryStore, llm: LLMClient,
                 reconciler: Reconciler, *, search_k: int = 5): ...
    async def add(self, messages: list[dict], scope: Scope) -> list[Memory]: ...  # extract → reconcile → dispatch
    async def search(self, query: str, scope: Scope, limit: int = 10) -> list[ScoredMemory]: ...
    async def get(self, id: str, scope: Scope) -> Memory | None: ...
    async def forget(self, id: str, scope: Scope) -> bool: ...

def build_memory() -> MemoryService: ...   # L4's construction entry point: wires store+embedder+llm+reconciler from settings
```

**`agent_id` v0 semantics:** `Scope` carries `agent_id` and the store writes it on every row, but
v0 search/get/delete filter on `user_id` (+ `namespace`) only — `agent_id` is not yet a query
filter. Named limitation, not a silent gap.

**`content_hash` policy:** stored on every row (SHA-256 hex of content), NOT a UNIQUE constraint —
L2's "duplicates allowed" stays literally true. Exact-dup handling is a reconciler decision (L3),
not a storage-layer rejection.

### 2.2 Embedder implementations

**CP-2 pinned values: model = `sentence-transformers/all-MiniLM-L6-v2`, D = 384.**

```python
# src/brain/embeddings.py

class Embedder(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

class SentenceTransformerEmbedder(Embedder):
    """Local free embedder. Downloads model once on first run; works offline after."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # lazy-load: import SentenceTransformer inside __init__ to avoid import-time cost
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    async def embed(self, text: str) -> list[float]:
        # SentenceTransformer.encode is sync; offload so the event loop is not blocked.
        vec = await asyncio.to_thread(self._model.encode, text, normalize_embeddings=True)
        return vec.tolist()

class FakeEmbedder(Embedder):
    """Deterministic, stable, semantically-meaningless vectors for unit tests.
    Produces a D=384 float vector derived from character ordinals of the text.
    Same input → same output always; different inputs → different outputs (mod wrap).
    NOT semantic — do NOT use to test recall quality; use to test ordering and isolation."""
    D = 384

    async def embed(self, text: str) -> list[float]:
        import math
        vec = [0.0] * self.D
        for i, ch in enumerate(text):
            vec[i % self.D] += ord(ch)
        # normalise to unit sphere so cosine scores are in [-1, 1]
        mag = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / mag for x in vec]
```

### 2.3 Schema DDL — `store/schema.sql`

Two tables: a regular companion table (`memories`) holding all text fields, and a vec0 virtual
table (`vec_memories`) holding the embedding plus the partition columns for pre-filter KNN.

**CP-1b pinned layout:**

```sql
-- store/schema.sql

CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    agent_id    TEXT,                        -- nullable; v0 not a query filter
    namespace   TEXT NOT NULL DEFAULT 'default',
    metadata    TEXT NOT NULL DEFAULT '{}',  -- JSON text
    content_hash TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_user_ns ON memories (user_id, namespace);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
    embedding   float[384],
    user_id     TEXT PARTITION KEY,
    namespace   TEXT PARTITION KEY
);
-- vec_memories.rowid is the join key to memories.rowid (they share the implicit rowid).
-- agent_id is NOT a partition key in vec0 because vec0 rejects NULL partition values;
-- it lives only in the companion table.
```

Note on rowid join: `memories` uses an implicit rowid (no `WITHOUT ROWID`). When inserting,
insert into `memories` first (get its `rowid` via `lastrowid`), then insert into `vec_memories`
with the same `rowid`. This is the stable join key between the two tables.

### 2.4 KNN SQL — pre-filter (CP-1a)

**The correct form — KNN entirely within the scope partition:**

```sql
-- Step 1: KNN within scope (pre-filter, executed inside vec_memories)
SELECT rowid, distance
FROM   vec_memories
WHERE  embedding MATCH ?          -- query vector as blob
  AND  k = ?                      -- limit
  AND  user_id = ?
  AND  namespace = ?
ORDER BY distance;                -- sqlite-vec returns cosine distance ascending

-- Step 2: fetch companion rows by rowid (one query with IN clause)
SELECT id, content, user_id, agent_id, namespace, metadata, content_hash, created_at, updated_at
FROM   memories
WHERE  rowid IN (/* rowids from step 1 */);
```

**FORBIDDEN form (post-filter — do NOT use):**

```sql
-- WRONG: global KNN then filter — another user's vectors crowd out alice's results
SELECT m.*, v.distance
FROM   vec_memories v
JOIN   memories m ON m.rowid = v.rowid
WHERE  v.embedding MATCH ?
  AND  v.k = ?
  AND  m.user_id = ?           -- <-- this is a post-filter, NOT a pre-filter
ORDER BY v.distance;
```

**Cosine score conversion:** sqlite-vec returns `distance` as cosine **distance** (0 = identical,
2 = opposite). Convert to similarity score for `ScoredMemory.score`:

```
score = 1.0 - distance
```

Range: [-1, 1]; higher is more similar. This matches the master plan's "higher-is-better" contract.

**Passing the query vector:** serialize the Python `list[float]` to a blob using
`sqlite_vec.serialize_float32(vec)` before binding.

### 2.5 `MemoryStore` ABC — `store/base.py`

Restated verbatim from §5.1; no additions, no renames.

```python
# src/brain/store/base.py
from abc import ABC, abstractmethod
from brain.models import Memory, ScoredMemory, Scope

class MemoryStore(ABC):
    @abstractmethod
    async def add(self, content: str, scope: Scope, metadata: dict | None = None) -> Memory: ...
    @abstractmethod
    async def search(self, query: str, scope: Scope, limit: int = 10) -> list[ScoredMemory]: ...
    @abstractmethod
    async def get(self, id: str, scope: Scope) -> Memory | None: ...
    @abstractmethod
    async def delete(self, id: str, scope: Scope) -> bool: ...
    @abstractmethod
    async def update(self, id: str, content: str, scope: Scope) -> Memory | None: ...
```

---

## 3. Implementation order

Each step is a working state. Do not skip steps; do not implement the next step until the current
one passes its own check.

**Step 1 — Project skeleton**
Create the directory tree (§1.6), `pyproject.toml` (§1.2), `.env.example` (§1.5).
Run `uv sync --extra dev`. Verify: `python -c "import sqlite_vec; print(sqlite_vec.__version__)"` prints `0.1.9`.

**Step 2 — `config.py`**
`Settings(BaseSettings)` with `brain_db_path: str = "./brain.db"` and
`brain_embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"`. Reads from `.env`.
Export a module-level `settings = Settings()`.

**Step 3 — `models.py`**
Paste the §5.1 types verbatim: `Scope`, `Memory`, `ScoredMemory`, `FactCandidate`,
`MemoryActionKind`, `MemoryAction`, `Reconciler`. Add necessary imports (`pydantic`, `abc`, `enum`).
`MemoryService` and `build_memory` are **stubs** (raise `NotImplementedError`) — present so other
modules can import without error.

**Step 4 — `embeddings.py`**
Implement `Embedder` ABC, `SentenceTransformerEmbedder`, and `FakeEmbedder` exactly as in §2.2.
Smoke-check: `python -c "import asyncio; from brain.embeddings import FakeEmbedder; v=asyncio.run(FakeEmbedder().embed('hello')); assert len(v)==384"`.

**Step 5 — `store/schema.sql` + schema application**
Write the DDL (§2.3). Add a `_apply_schema(db)` helper in `sqlite.py` that reads and executes
`schema.sql` (using `executescript`). The helper is called once at store construction time inside
a `to_thread` call.

**Step 6 — `store/base.py`**
`MemoryStore` ABC, verbatim (§2.5).

**Step 7 — `store/sqlite.py` — `SQLiteMemoryStore`**
Constructor pattern: provide an async classmethod `create(db_path, embedder)` so tests can
`await SQLiteMemoryStore.create(path, embedder)` without needing a running loop at `__init__` time.
`create` calls `asyncio.to_thread(_apply_schema, db_path)` to initialize the schema.

Implement all 5 methods. Each method:
1. Awaits `asyncio.to_thread(_work, ...)` where `_work` is a sync function.
2. `_work` opens a connection via `_open_db(path)`, does its work, closes the connection, returns
   plain Python objects (no `sqlite3.Row` crosses the thread boundary).
3. The async method converts the return value to the appropriate `Memory` / `ScoredMemory` / `bool`.

`add`:
- Generate `id = str(uuid.uuid4())`.
- Compute `content_hash = hashlib.sha256(content.encode()).hexdigest()`.
- Embed content (async, before entering thread): `embedding = await self._embedder.embed(content)`.
- In thread: INSERT into `memories` (capture `lastrowid`), then INSERT into `vec_memories` with
  same rowid and `sqlite_vec.serialize_float32(embedding)`.
- Return `Memory(...)`.

`search`:
- Embed query (async, before thread): `q_vec = await self._embedder.embed(query)`.
- In thread: Step 1 KNN (§2.4) returns `[(rowid, distance), ...]`. Step 2 fetches companion rows
  by rowid. Re-sort by distance ascending (Step 1 order may not be preserved by Step 2 fetch).
- Return `list[ScoredMemory]` with `score = 1.0 - distance`, ordered highest-score-first.

`get`:
- In thread: `SELECT ... FROM memories WHERE id = ? AND user_id = ? AND namespace = ?`.
- Returns `Memory | None`.

`delete`:
- In thread: look up `rowid` via `SELECT rowid FROM memories WHERE id=? AND user_id=? AND namespace=?`.
  If not found, return `False`. DELETE from `vec_memories WHERE rowid=?`, then
  DELETE from `memories WHERE rowid=?`. Return `True`.

`update`:
- `get` the existing row (async). If not found, return `None`.
- Re-embed new content (async).
- In thread: fetch `rowid` from `memories WHERE id=?`; UPDATE `memories` SET `content=?,
  content_hash=?, updated_at=?` WHERE `rowid=?`; UPDATE `vec_memories` SET `embedding=?`
  WHERE `rowid=?`.
- Return updated `Memory`.

**Step 8 — `scripts/demo_layer1.py`**
Inserts 5 alice facts and 2 bob facts (see §5.2), queries `"what food do I like"` as alice,
prints results with scores. Bob's rows must never appear. Script must exit with code 0.

**Step 9 — `tests/conftest.py` + `tests/test_store.py`**
Write fixtures and all 8 tests (§4).

---

## 4. Test fixtures

### 4.1 `tests/conftest.py`

```python
import pytest
import pytest_asyncio
from pathlib import Path
from brain.embeddings import FakeEmbedder
from brain.store.sqlite import SQLiteMemoryStore

@pytest.fixture
def fake_embedder():
    return FakeEmbedder()

@pytest_asyncio.fixture
async def store(tmp_path: Path, fake_embedder):
    db_path = str(tmp_path / "test_brain.db")   # REAL temp file — :memory: is FORBIDDEN
    s = await SQLiteMemoryStore.create(db_path, fake_embedder)
    yield s
```

**`:memory:` is FORBIDDEN** — sqlite-vec's vec0 virtual table does not reliably support
`:memory:` databases across all versions; always use a real temp file via pytest's `tmp_path`.

### 4.2 `tests/test_store.py` — the oracle

Tests must use `FakeEmbedder` (deterministic) and assert exact ordering and counts.

**Test groups:**

1. **`test_add_returns_memory`** — `add` returns a `Memory` with a non-empty `str` id, correct
   `content`, correct `user_id`, `namespace="default"`, a non-empty `content_hash`, and timestamps
   that are non-empty ISO-8601 strings.

2. **`test_search_scope_isolation`** — insert 2 alice facts + 1 bob fact; search as alice; assert
   `len(results) == 2` and every `result.memory.user_id == "alice"`. Bob's row must never appear.

3. **`test_search_ordering`** — insert at least 3 alice facts with varied text; search with a
   query; assert results are ordered highest-score-first:
   `all(results[i].score >= results[i+1].score for i in range(len(results)-1))`.

4. **`test_get_found_and_not_found`** — `get` returns the correct `Memory` when it exists, `None`
   when the id does not exist, and `None` when the id belongs to a different user.

5. **`test_delete`** — `add` a fact, `delete` it (assert returns `True`), then `get` returns
   `None`. Assert `delete` on a non-existent id returns `False`.

6. **`test_update`** — `add` a fact, `update` with new content; assert returned `Memory` has the
   new `content`, new `content_hash`, `updated_at >= created_at`. Confirm via `get` that the stored
   record reflects the update.

7. **`test_namespace_isolation`** — insert same `user_id` in two different namespaces; search in
   one namespace; assert only rows from that namespace are returned.

8. **`test_live_embedder_smoke`** (single live test, no key) — construct a
   `SentenceTransformerEmbedder`, call `embed("hello world")`, assert `len(vec) == 384` and
   `isinstance(vec[0], float)`. Does NOT assert specific vector values. Mark with
   `@pytest.mark.slow` so it can be excluded from fast CI runs if desired.

---

## 5. Pinned values and assertions

### 5.1 Checkpoint summary — do not regress

| Checkpoint | Value |
|---|---|
| **CP-2 embedder model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **CP-2 D** | **384** — `float[384]` in DDL, `FakeEmbedder.D = 384`, live smoke asserts `len==384` |
| **CP-1 sqlite-vec pip version** | **`sqlite-vec==0.1.9`** |
| **CP-1 load sequence** | `db.enable_load_extension(True)` → `sqlite_vec.load(db)` → `db.enable_load_extension(False)` |
| **CP-1a KNN form** | Pre-filter: `WHERE embedding MATCH ? AND k = ? AND user_id = ? AND namespace = ?` entirely within `vec_memories`; post-filter JOIN is FORBIDDEN |
| **CP-1b table layout** | Two tables: `memories` (companion, implicit rowid) + `vec_memories` (vec0, embedding + PARTITION KEYs); `agent_id` in `memories` ONLY |
| **CP-1c connection strategy** | Connection-per-operation; `isolation_level=None` (autocommit); opened + closed on same `to_thread` worker; `check_same_thread` default; extension loaded per connection |
| **CP-1d test DB** | Real temp file via `tmp_path`; `:memory:` FORBIDDEN |

### 5.2 Demo fixture facts

**Alice's facts (5):**
1. `"Alice loves Italian food, especially pasta and risotto."`
2. `"Alice is allergic to shellfish."`
3. `"Alice's favourite dessert is tiramisu."`
4. `"Alice drinks oat milk, not cow's milk."`
5. `"Alice eats lunch at noon every weekday."`

**Bob's facts (2):**
1. `"Bob prefers Japanese food and sushi."`
2. `"Bob is vegetarian."`

**Query (as alice):** `"what food do I like"`

**Expected:** results ranked by cosine similarity descending (highest score first), all with
`memory.user_id == "alice"`, zero bob rows.

### 5.3 Cosine score formula

```
score = 1.0 - distance     # where distance is sqlite-vec's cosine distance output
```

`distance = 0.0` → `score = 1.0` (identical). `distance = 2.0` → `score = -1.0` (opposite).
Applied in `search` before constructing `ScoredMemory`.

### 5.4 `id` type invariant

Every `id` field on every `Memory` is `str(uuid.uuid4())` — a Python `str`. No `uuid.UUID` object
crosses any store/facade/tool boundary. Stored in a `TEXT` column.

### 5.5 Vector serialization

```python
import sqlite_vec
blob = sqlite_vec.serialize_float32(embedding)   # list[float] → bytes
```

Bind this blob to the `embedding MATCH ?` parameter and to the INSERT into `vec_memories`.

---

## 6. Layer completion verification

**Done when:**

```
uv run python scripts/demo_layer1.py
```

**Expected concrete output (structure; exact scores vary by embedder):**

```
Querying as alice: "what food do I like"
Results:
  1. score=0.XX  "Alice loves Italian food, especially pasta and risotto."
  2. score=0.XX  "Alice's favourite dessert is tiramisu."
  ...  (alice rows only; up to 5 results)

Bob's rows when searched as bob: 2 result(s) (scope isolation confirmed)
Alice rows never appear in bob's search.
```

**All assertions for "done":**
- Every printed alice-search result has `memory.user_id == "alice"` — zero bob rows.
- Results printed in descending score order (`score[i] >= score[i+1]`).
- `uv run pytest tests/test_store.py -v` passes all 8 tests with zero failures.
- No Docker, no API key required to run either command.

**Hands off to Layer 2:** the `MemoryStore` ABC (5 methods, each taking `scope: Scope`) + the
`Embedder` ABC + `SQLiteMemoryStore` (the concrete impl) + `FakeEmbedder` + the `Memory` /
`ScoredMemory` / `Scope` data contracts from `models.py`.

Layer 2 imports from `brain.store.base`, `brain.models`, and `brain.embeddings`. It does NOT
import from `brain.store.sqlite` directly (goes through the interface).
