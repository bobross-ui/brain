# Brain — Agentic Memory Layer — Implementation Plan (v0)

> **Master architecture plan.** Locks decisions so each layer can be built independently.
> Every layer is a *testable completion unit*: when it's done, the system-so-far runs
> end-to-end and its slice is observable, not just unit-tested.
>
> Goal: a **simple memory layer up and running quickly** — zero external infra, no API keys to
> start — that is *structured so future refinements are swap-an-implementation, not
> rewrite-the-whole-thing*. The seams (`Embedder`, `LLMClient`, `MemoryStore`) are where the
> growth happens.
>
> This file is **contracts, not code** — decisions, boundaries, and the shared data model.
> Concrete type definitions, signatures, SQL, and commands live in the `LAYER{N}_PLAN.md` files.

---

## 0. Setup posture — no blockers to start (replaces the old provider-key gate)

The v0 path needs **no API key and no Docker**:

- **Embedder:** local & free — runs on CPU, downloads its model **once** on first run, then works
  offline. No key, no per-call cost.
- **Storage:** a local **SQLite** file via the **sqlite-vec** extension — no server, no container.
- The only later external dependency is the **L2+ LLM** (fact extraction / reconciliation). Its
  free/local provider is resolved at the **L2 checkpoint (CP-3)** and **does not block L1**.

→ Build Layer 1 and you immediately have a runnable, queryable memory store. That *is* the quick v0.

---

## 1. Locked decisions

| Concern | Choice | Why (short) |
|---|---|---|
| Language / runtime | **Python 3.11+** | Dominant agent/AI ecosystem; best embedding/LLM/DB tooling. |
| Memory model | **Hybrid: facts + semantic** | LLM-distilled atomic facts, stored for semantic recall (mem0-style). Core IP. |
| Interface | **MCP server (stdio)** | Any MCP agent/client plugs in via `remember`/`recall`/`forget`. This is the point of "agent use." |
| Storage | **SQLite + sqlite-vec, behind a `MemoryStore` interface** | Zero infra (no Docker/server) → *up and running quick*. Swappable to Postgres/pgvector later **through the interface** — an impl swap, not a rewrite. **Chosen for light infra, NOT cost** (both are $0). |
| Embedder | **Local/free embedder behind the `Embedder` interface**; exact model + dim pinned at L1 | No key, no per-call cost, CPU-friendly. Swappable to a hosted embedder later behind the same interface. |
| LLM | **`LLMClient` interface; free/local default, provider resolved at L2 (CP-3)** | Extraction/reconciliation need an LLM but only from L2. Keep the seam now; defer the provider choice. |
| Distance metric | **Cosine similarity** — score higher-is-better, range [-1, 1] | One metric across L1's index/search and L3's candidate threshold; L1 pins the matching sqlite-vec distance function. Without this fixed, CP-4's threshold has no meaning. |
| Reconciler seam | **`Reconciler` contract; "always-add" default in L2, LLM-backed in L3** | Lets L3 be purely *additive* (swap an impl) instead of editing L2's write-loop. |
| Concurrency | **Async `MemoryStore` interface; sync SQLite calls offloaded via `asyncio.to_thread`** | The MCP SDK is async. Keeping the store interface async means MCP **and** a future asyncpg swap stay clean — no sync↔async impedance mismatch at L4. The SQLite driver is sync, so we *wrap* it; we do not fake an async-native driver. Locked once, here. |
| Scoping | **`user_id` + `agent_id` + `namespace` in the schema from day one** | Multi-tenant is the most painful thing to retrofit. Columns present even if only `user_id` is exercised early. |
| Swap seams | **`Embedder` + `LLMClient` + `MemoryStore` interfaces** | The three places future refinement happens. No layer imports `sqlite-vec` or a model SDK directly except behind these. |
| Config | **`pydantic-settings` + `.env`** | One typed settings object; DB path / model names / provider out of code. |
| Package/dep mgmt | **`uv` + `pyproject.toml`** | Fast, reproducible; lockfile committed. |

---

## 2. Open checkpoints

> **Do NOT resolve these here.** Each layer plan resolves its own with verified, current
> information (package versions, model dims, and SDK schemas drift).

- ⚠️ **CP-1 (L1)** — **sqlite-vec provisioning + table layout + scope-filtered KNN + connection
  strategy.** Pinned mechanism: the **`sqlite-vec`** pip package, loaded via
  `enable_load_extension` + `sqlite_vec.load`. Resolve, **in this order**:
  - (a) **Scope-filtered KNN — resolve FIRST, highest-risk.** The scope filter (`user_id` [+
    `agent_id` + `namespace`]) must be applied as a **PRE-filter**, i.e. the top-`k` is computed
    *within* the scope. A **post-filter** (take global top-`k`, then drop non-matching rows) is
    **wrong** — another user's vectors can crowd out the top-`k` so alice silently gets fewer than
    `limit` (or near-empty) results even though her memories exist. This *looks* correct on tiny
    fixtures and fails at real size. Verify the current sqlite-vec version's support for in-query
    metadata/partition filtering on a KNN search; if inadequate, the companion-table/join approach
    must still produce **correct top-`k` within scope**, never a post-filter truncation.
  - (b) **Table layout** — how the Memory record maps onto sqlite-vec: a `vec0` virtual table for
    the embedding plus scope/metadata columns (auxiliary/partition/metadata columns vs. a companion
    regular table joined on `id`). Must support (a).
  - (c) **Connection / threading strategy** — because the async interface offloads sync SQLite via
    `asyncio.to_thread`, calls run on arbitrary pool threads. A SQLite connection can't cross
    threads by default (`check_same_thread`), and the sqlite-vec extension must be loaded **per
    connection**. Pick deliberately (connection-per-operation, a small pool, or a single dedicated
    worker thread) so there is no "SQLite objects created in a thread can only be used in that same
    thread" hazard.
  - (d) The test-DB mechanism (temp file vs. `:memory:`). Verify the current sqlite-vec version + API.
- ⚠️ **CP-2 (L1)** — **Exact local embedder model + vector dimension `D`.** Default candidate: a
  small `sentence-transformers` model (e.g. `all-MiniLM-L6-v2`, D=384). Verify current package +
  model + dim. Blocks the fixed-width vector(`D`) column.
- ⚠️ **CP-3 (L2)** — **Free/local LLM** provider+model for fact **extraction** (e.g. Ollama + a
  small instruct model) + the extraction prompt + the output JSON schema (messages → list of
  atomic facts).
- ⚠️ **CP-4 (L3)** — **Reconciliation** prompt + decision schema (ADD/UPDATE/DELETE/NOOP) + the
  **cosine-similarity** threshold and `k` for fetching candidate existing memories.
- ⚠️ **CP-5 (L4)** — **MCP Python SDK** version + transport (**stdio** default) + the exact tool
  input/output schemas for `remember` / `recall` / `forget`.

---

## 3. The 4 layers (what "done" means per layer)

> **Layer boundary rule:** a layer is not done until you can run it as a user would and observe
> correct output for that slice. Every "done when" below is a *runnable demo*, observable under
> LLM nondeterminism (assert structure, not exact model strings, on live paths — see §6).

### Layer 1 — Storage + Retrieval foundation (the "dumb" store) — **the quick v0**
- **Scope:** the SQLite + sqlite-vec schema (with scoping columns), the `Embedder` interface +
  default local impl + a deterministic fake embedder for tests, the **`MemoryStore` interface (all
  five methods)**, and `SQLiteMemoryStore` implementing add / search (vector similarity + scope
  filter) / get / delete / **update** (update-in-place: re-embed new content + bump `updated_at`).
  `update` is implemented and unit-tested here so the seam is complete from day one, but **no write
  path calls it yet** — L3 wires it in. **No LLM extraction or reconciliation.**
- **Done when:** `uv run python scripts/demo_layer1.py` (no Docker, no key) inserts ~5 facts for
  `user="alice"` and 2 for `user="bob"`, queries *"what food do I like"* as alice, and prints
  results ranked by similarity with **bob's memories never returned**.
- **Hands off to Layer 2:** the async `MemoryStore` (add/search/get/delete/update — `add` takes
  content + scope + metadata, returns the stored record; `search` takes a query string + scope +
  limit, returns memories scored by cosine similarity; `update` re-embeds + bumps `updated_at`),
  the `Embedder` interface, and the Memory record contract (§5).

### Layer 2 — Extraction + naive add (messages → facts → store)
- **Scope:** an extraction step (conversation messages → atomic fact candidates) via `LLMClient`;
  the `Reconciler` contract with a default "always-add" implementation; and a write path that
  distills facts and runs each candidate through the active reconciler before storing. With the
  default reconciler this is naive add — **duplicates allowed, do NOT dedup or update.** The
  reconciler seam is the boundary that keeps L3 additive, not scope creep.
- **Done when:** a demo feeds a fixture multi-turn conversation; with the **recorded LLM response
  fixture** it stores the exact expected fact set (assert count + each fact present); the live
  smoke run stores ≥1 structurally-valid fact.
- **Hands off to Layer 3:** the extraction step, the fact-candidate contract, and the
  **`Reconciler` contract** (plus the store's scope-restricted candidate search). L3 supplies a new
  reconciler implementation and swaps it in — **L3 does not edit L2's write-loop.**

### Layer 3 — Reconciliation (ADD / UPDATE / DELETE / NOOP) — core IP
- **Scope:** implement the LLM-backed reconciler — a near-pure decision (a fact candidate + a list
  of similar existing memories → one action) — and wire it in as the active reconciler (replacing
  the default). The write path fetches similar existing memories via the store's scope-restricted
  cosine search, then applies the returned action (insert / update-in-place / delete / skip) by
  calling the store's **existing** methods — `add` / `update` (defined & implemented in L1) /
  `delete` / no-op. L3 adds no new store method; it only **wires** reconciliation into the write
  path. **Purely additive over L2.**
- **Done when:** a demo seeds *"User likes pizza"*, then feeds a conversation containing *"Actually
  I prefer sushi now"*; with the recorded action fixture the pizza memory is **UPDATED in place (no
  contradictory duplicate row)** — assert the action sequence and final store state. The
  deterministic UPDATE/DELETE/NOOP paths are proven by **unit-testing the reconcile decision with
  injected candidate lists** (§6), not by relying on the embedder to place "pizza" near "sushi."
- **Hands off to Layer 4:** the complete async `MemoryService` facade (§5.1) — add(messages, scope),
  search(query, scope), get(id, scope), forget(id, scope) — plus the `build_memory()` factory.

### Layer 4 — MCP server
- **Scope:** wrap the L3 `MemoryService` facade (constructed via `build_memory()`) as MCP tools
  (`remember`, `recall`, `forget`) over **stdio** transport. **No auth, no extra transports**
  (HTTP/SSE is future).
- **Done when:** launch the server and, from an MCP client/inspector (or the in-process test
  client), call `remember` with a conversation, then in a *separate* call `recall` a query and
  observe the persisted memory returned across calls (structural assertion).
- **Hands off to:** the product — a runnable MCP memory server any agent can connect to.

---

## 4. Repository layout

```
brain/
  pyproject.toml            # uv-managed; deps + tool config
  uv.lock
  .env.example              # DB path, embedder/LLM model names, LLM provider (L2+)
  README.md
  src/brain/
    __init__.py
    config.py               # settings: DB path, model names, provider
    models.py               # Memory, ScoredMemory, FactCandidate, MemoryAction, Reconciler
    embeddings.py           # Embedder interface; local default impl; fake embedder (tests)
    llm.py                  # LLMClient interface; default impl (L2)
    store/
      __init__.py
      base.py               # MemoryStore interface — the swap seam (L1)
      schema.sql            # sqlite-vec table(s) + indexes + scoping columns
      sqlite.py             # SQLiteMemoryStore: add/search/get/delete (+ update from L3)
    extract.py              # L2: messages -> fact candidates
    reconcile.py            # L2 default reconciler + L3 LLM reconciler
    memory.py               # MemoryService facade (§5.1) + build_memory() factory
    mcp_server.py           # L4: MCP tools over stdio
  scripts/
    demo_layer1.py
    demo_layer2.py
    demo_layer3.py
  tests/
    conftest.py             # temp sqlite-vec DB fixture, fake embedder, recorded LLM responses
    fixtures/
      conversations/        # input message fixtures
      llm_responses/        # recorded extraction + reconciliation responses (the oracle)
    test_store.py           # L1
    test_extract.py         # L2
    test_reconcile.py       # L3
    test_mcp.py             # L4
```

> Note vs. a Postgres build: **no `docker-compose.yml`**; `store/base.py` holds the `MemoryStore`
> interface (the seam a future `postgres.py` would also implement); `store/sqlite.py` replaces
> `postgres.py`.

---

## 5. Cross-cutting concerns (every layer obeys these)

**Shared contracts — owned by L1, honored by every layer.** These are *contracts* (field +
type/semantics), not code: L1 writes the actual class/migration, but the shape below is fixed here
so the parallel layer sessions all agree on it. **Do NOT defer these to a layer plan.**

**Memory record:**

| Field | Type / semantics | Notes |
|---|---|---|
| `id` | **`str`** = `str(uuid.uuid4())` | primary key; TEXT column; **always a Python `str`, never a `uuid.UUID` object** across any store/facade/tool boundary |
| `content` | text | the atomic fact |
| `embedding` | float32 vector, dim `D` | `D` = embedding dim, pinned by CP-2; stored via sqlite-vec; cosine distance |
| `user_id` | text, **required** | scope — never global |
| `agent_id` | text, **nullable** | optional sub-scope |
| `namespace` | text, default `"default"` | scope partition |
| `metadata` | JSON object | free-form (source, tags, …) |
| `content_hash` | text | exact-dup signal; **NOT** a UNIQUE constraint (policy below) |
| `created_at` / `updated_at` | timestamp, ISO-8601 UTC text | set on write / on update-in-place |

### 5.1 Pinned shared types — restate these VERBATIM; conform call sites; change NOTHING

> These are the cross-layer contracts. Any layer that touches one **restates it exactly** and
> adapts its own call sites to match — **no layer redefines, renames, "promotes", or re-types any
> of them.** `pydantic.BaseModel` for data types; `abc.ABC` for seams. Modules are noted in
> comments. If a fragment here disagrees with anything in a layer plan, **this section wins.**

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

**`MemoryAction` is structured from L2 onward.** L2 defines it, `AlwaysAddReconciler` returns it,
and L2's write-loop dispatches on `action.kind`. **L3 does NOT redefine or "promote" it** — L3 only
*produces* it from the LLM decision and fills the `_apply` UPDATE/DELETE branches. No bare-`Enum`
stage exists at any point.

**`agent_id` v0 semantics.** `Scope` carries `agent_id` and the store writes it on every row, but
**v0 search/get/delete filter on `user_id` (+ `namespace` where the store partitions on it) only** —
`agent_id` is not yet a query filter. A deliberate, documented limitation (named future refinement),
not a silent gap.

**Reconciler / facade swap.** Only the `Reconciler` injected into `MemoryService` changes between L2
(`AlwaysAddReconciler`) and L3 (`LLMReconciler`). The write-loop, the store, and the facade
signatures never change. L4 constructs via `build_memory()` and calls only the four facade methods.

**`content_hash` policy.** Stored on every row for future exact-dup detection, but **NOT a DB
`UNIQUE` constraint** — so L2's "duplicates allowed" stays literally true. Exact-dup handling, if
ever wanted, is a reconciler decision, not a storage-layer rejection.

**Async everywhere (via wrapping).** The `MemoryStore`, `Embedder`, and `LLMClient` expose async
methods; MCP handlers are async. The sync SQLite driver is called inside the store via
`asyncio.to_thread` (or equivalent) — **no raw blocking DB call on the event loop**, and no sync
public method on these interfaces.

**Scoping.** Reconciliation and search operate *within a scope* (`user_id` [+ `agent_id` +
`namespace`]). All store reads/writes take the scope explicitly — never global.

**Provider/storage abstraction.** Everything that touches a model goes through `Embedder` /
`LLMClient`; everything that touches the database goes through `MemoryStore`. No other module
imports a model SDK or `sqlite-vec` directly.

**Config.** One settings object from `pydantic-settings`; nothing reads the environment directly.
`.env.example` lists every var (DB path, embedder/LLM model names, LLM provider). No key is
required for the L1 path.

---

## 6. Testing strategy

**Provisioning (CP-1, cross-cutting — pinned here):** tests run against a **real sqlite-vec
database** — a temp SQLite file (or `:memory:` with the extension loaded), resolved at CP-1. **No
mocking of the DB**; vector ordering and scope filtering must be tested for real. No Docker. Each
layer reuses this fixture.

**LLM/embedding nondeterminism contract (the highest-risk assumption):**
- **Deterministic path (the oracle):** a **fake embedder** (stable, *lexically-derived* vectors
  from text — semantically meaningless on purpose) + **recorded LLM responses** in
  `tests/fixtures/llm_responses/`. These back **exact** assertions — fact sets in L2, the action
  sequence + final store state in L3. The fake embedder's stable-but-not-semantic vectors are what
  let L3's UPDATE/DELETE/NOOP paths be reached via **injected candidate lists** instead of hoping
  the embedder places two phrases near each other.
- **Live path:** exactly **one** smoke test per layer hitting the real model. For the **embedder**,
  "live" is local, free, and deterministic (no key) — still assert structure only, not exact
  vectors. For the **LLM** (L2/L3), never assert exact model output against a live call; loose
  structural assertions only (e.g. "≥1 fact extracted").

Per-layer: L1 asserts similarity ordering + scope isolation with the fake embedder; L2 asserts fact
extraction against recorded responses; **L3 unit-tests the reconcile decision (a candidate + a list
of existing memories → an action) with injected candidate lists**, plus an integration test for "no
contradictory duplicate row"; L4 asserts round-trip persistence across separate tool calls.

---

## 7. Tradeoffs (named explicitly — choices NOT made)

- **SQLite + sqlite-vec over Postgres + pgvector:** zero infra now (no Docker) → fastest path to a
  runnable slice; behind the `MemoryStore` interface the swap to pgvector later is an impl change,
  not a rewrite — the only migration tax is re-embedding into a fixed-width column. *(Chosen for
  light infra / quick start — **not** for cost; both options are $0.)*
- **Local free embedder over a hosted embedder:** no key, no per-call cost, runs on CPU; recall
  quality is below large hosted embedders — swappable later behind `Embedder`.
- **Async store interface wrapping sync SQLite (via `asyncio.to_thread`) over a sync-native store:**
  small wrapping overhead now, but MCP (L4) and a future asyncpg swap stay clean — no sync↔async
  rewrite later.
- **LLM fact extraction on the write path (L2+) over store-everything:** better recall quality and a
  true "facts" layer; the (free/local) LLM call is the price; deferred to L2.
- **Synchronous reconciliation on write over background reconciliation:** simpler and observable
  now; can move to a background queue later.
- **Dense-vector retrieval only:** no hybrid keyword/BM25 search and no reranking yet — deferred to
  a future "advanced retrieval" layer.
- **Scoping columns present but only `user_id` exercised initially:** small schema cost now to avoid
  a multi-tenant migration later.
- **stdio-only MCP transport:** HTTP/SSE + auth deferred.

---

## 8. Future layers / refinements (the growth path — out of scope for v0)

Postgres/pgvector swap for production scale (drop-in via `MemoryStore`) · hosted embedder/LLM option
(drop-in via `Embedder`/`LLMClient`) · advanced retrieval (hybrid + rerank, recency/decay
weighting) · background/async reconciliation queue · HTTP/SSE MCP transport + auth/multi-tenant API
keys · memory summarization/compaction · graph relations between facts · eval harness for recall
quality.
