# Brain

Local-first agentic memory layer for agents. The project stores scoped memories and raw conversation
turns in SQLite, retrieves them with hybrid vector + FTS5/BM25 search, extracts and reconciles atomic
facts, and exposes the memory service as an MCP stdio server.

## Status

| Layer | Status | What works now |
| --- | --- | --- |
| Layer 1: Storage + Retrieval | Complete | SQLite + sqlite-vec memories, append-only raw sessions/turns, hybrid vector + FTS5/BM25 retrieval, RRF fusion, subject filters, scoped CRUD |
| Layer 2: Extraction + Naive Add | Complete | Ollama-backed fact extraction, fake LLM fixture path, always-add reconciler, MemoryService facade, Layer 2 demo |
| Layer 3: Reconciliation | Complete | LLM-backed ADD/UPDATE/DELETE/NOOP reconciler, candidate thresholding, update-in-place path, Layer 3 demo |
| Layer 4: MCP Server | Complete | `remember`, `recall`, `recall_evidence`, and `forget` MCP tools over stdio |

Current behavior: raw sessions and turns are stored before extraction. Messages are then extracted
into atomic facts, vector-only similar memories are retrieved for reconciliation, and the active
reconciler decides whether to add, update, delete, or skip each fact. User recall uses reciprocal
rank fusion (RRF) over sqlite-vec and memory FTS5/BM25 candidates. The unified evidence path fuses
those memory hits with raw-turn BM25 hits for answering and evaluation. Session ingestion is
idempotent and retry-safe. The default factory wires `LLMReconciler` with the configured LLM
backend, and the MCP server wraps that service over stdio. Auth and HTTP transport are not built.

## Architecture

The core seams are:

- `Embedder`: converts text into vectors and reports its `dim`, which sets the sqlite-vec
  column width at schema-creation time. Current implementation: `SentenceTransformerEmbedder`.
- `MemoryStore`: async storage interface. Current implementation: `SQLiteMemoryStore`.
- `LLMClient`: async JSON chat interface. Current implementations: `OllamaLLMClient` and
  `DeepSeekLLMClient`.
- `Reconciler`: decides what to do with extracted facts. Current implementations:
  `AlwaysAddReconciler` and `LLMReconciler`.
- `MemoryService`: facade used by later layers: `add`, `ingest_session`, `search`,
  `recall_evidence`, `search_turns`, `get`, and `forget`.

Memory scope is always explicit through `Scope(user_id, agent_id, namespace)`. Search and read paths
filter by `user_id` and `namespace`.

## Requirements

- Python 3.11+
- `uv`
- macOS/Linux with local SQLite extension and FTS5 support
- Ollama for the default local backend, live local tests, demos, and MCP `remember` calls

Install `uv` on macOS:

```bash
brew install uv
```

Install Python dependencies:

```bash
uv sync --extra dev
```

## Configuration

Start from the example env file:

```bash
cp .env.example .env
```

Available settings:

```bash
BRAIN_DB_PATH=./brain.db
BRAIN_EMBEDDER_MODEL=sentence-transformers/all-MiniLM-L6-v2
BRAIN_RERANKER_MODEL=
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:3b-instruct
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

No API keys are required for the default local (Ollama) backend.

`BRAIN_RERANKER_MODEL` is optional and disabled by default. Set it to a local
sentence-transformers cross-encoder model to rerank the fused candidate pool. The model is loaded
on first use and may be downloaded by sentence-transformers.

## LLM Backends

The `LLMClient` seam used for fact extraction and reconciliation has two implementations,
selected by `LLM_PROVIDER`:

- `ollama` (default): local, free, no API key. Uses `LLM_MODEL` (e.g. `qwen2.5:3b-instruct`).
- `deepseek`: the DeepSeek API (OpenAI-compatible). Set:

  ```bash
  LLM_PROVIDER=deepseek
  LLM_MODEL=deepseek-v4-pro   # or deepseek-v4-flash for a lighter/cheaper model
  DEEPSEEK_API_KEY=sk-...
  ```

DeepSeek uses JSON output mode (`response_format={"type":"json_object"}`); the requested
schema is injected into the prompt as guidance since DeepSeek does not enforce a JSON schema.
Embeddings always stay local via `SentenceTransformerEmbedder` (DeepSeek has no embedding
endpoint), so only extraction/reconciliation/answering move to the API. The provider factory
is `brain.llm.build_llm_client(provider, model)`.

## Embedding Model

Embeddings are produced locally by `SentenceTransformerEmbedder` and selected with
`BRAIN_EMBEDDER_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`, 384-dim). The
sqlite-vec column width is derived from the embedder's `dim` and templated into the initial
schema migration at database-creation time, so any model works without code changes.

Because `CREATE VIRTUAL TABLE` fixes the vector dimension at creation, switching to a
**different-dimension** model requires rebuilding the database — delete `BRAIN_DB_PATH` and let
it recreate. Same-dimension swaps only need a re-embed of existing rows.

The store embeds queries and documents through the same call, so prefer **symmetric** models:
`thenlper/gte-base` (768-dim) is a strong drop-in upgrade over the default. Asymmetric models
that expect a query-only instruction prefix (`bge-*`, `mxbai-embed-large-v1`, `e5-*`) would
need that prefix added at the embed sites to reach their full retrieval quality.

## Ollama Setup

Layer 2 live extraction and Layer 3 live reconciliation use Ollama with `qwen2.5:3b-instruct`.

Recommended macOS install:

```bash
brew install --cask ollama-app
ollama --version
ollama pull qwen2.5:3b-instruct
ollama run qwen2.5:3b-instruct "Reply only: ok"
```

If `ollama run` fails with `llama-server binary not found`, remove the Homebrew formula and use the
cask:

```bash
brew uninstall ollama
brew reinstall --cask ollama-app
```

For live tests or demos, keep Ollama running in a separate terminal:

```bash
ollama serve
```

Stop it when done with `Ctrl-C`, or kill any leftover server process:

```bash
pgrep -fl "ollama serve"
kill <pid>
```

## Run

Layer 1 storage/retrieval demo:

```bash
uv run python scripts/demo_layer1.py
```

Layer 2 deterministic fixture demo. This does not require Ollama:

```bash
uv run python scripts/demo_layer2.py --mode fixture
```

Layer 2 live demo. This requires `ollama serve` running:

```bash
uv run python scripts/demo_layer2.py --mode live
```

Layer 3 deterministic reconciliation demo. This does not require Ollama:

```bash
uv run python scripts/demo_layer3.py
```

Layer 3 live smoke demo. This requires `ollama serve` running and may produce nondeterministic
ADD/UPDATE/NOOP behavior:

```bash
uv run python scripts/demo_layer3.py --mode live
```

MCP stdio server. This blocks on stdio for an MCP client to connect:

```bash
uv run python -m brain.mcp_server
```

The server exposes:

- `remember(messages, user_id, agent_id=None, namespace="default")`
- `recall(query, user_id, agent_id=None, namespace="default", limit=10, filters=None)`
- `recall_evidence(query, user_id, agent_id=None, namespace="default", limit=10, filters=None)`
- `forget(id, user_id, agent_id=None, namespace="default")`

At this phase, `filters` supports `{"subject": "Caroline"}`. `recall` preserves the existing
`ScoredMemory` response shape; `recall_evidence` returns unified memory and raw-turn evidence.

## Tests

Run the full test suite:

```bash
uv run pytest
```

Live-smoke tests carrying the `ollama` marker are pinned to the Ollama backend: they run only
when `LLM_PROVIDER=ollama` and skip otherwise (e.g. on the DeepSeek backend).

Run deterministic Layer 2 tests:

```bash
uv run pytest tests/test_extract.py -k "not ollama" -v
```

Run deterministic Layer 3 tests:

```bash
uv run pytest tests/test_reconcile.py -v -m "not live"
```

Run non-live tests for completed layers:

```bash
uv run pytest -m "not live" -k "not slow and not ollama" -v
```

Run Layer 4 MCP transport tests:

```bash
uv run pytest tests/test_mcp.py -v
```

Run live Layer 2 Ollama smoke tests:

```bash
ollama serve
uv run pytest tests/test_extract.py -k "ollama" -v -s
```

Run the live Layer 3 Ollama smoke test:

```bash
ollama serve
uv run pytest tests/test_reconcile.py -m live -v -s
```

Run the live local embedder smoke test. This may download the sentence-transformers model on first
run:

```bash
uv run pytest tests/test_store.py -k "live_embedder" -v -s
```

## Evaluation (LOCOMO)

`scripts/eval_locomo.py` benchmarks Brain on the LOCOMO long-conversation dataset. It ingests each
conversation session-by-session via `MemoryService.ingest_session()` while preserving session IDs,
turn IDs, speakers, and dates. With `--retrieve-from memories`, it retrieves via
`MemoryService.recall_evidence()`, fusing hybrid memory hits with raw-turn hits, answers from the
unified evidence, and optionally judges correctness.
With `--retrieve-from turns`, it retrieves raw turns through FTS5/BM25 and reports retrieval recall
only.

Both modes report true evidence-ID recall when the dataset supplies gold turn IDs, plus the original
token-overlap recall proxy, and write per-question records to JSONL.

Get the dataset (`locomo10.json`) from the LOCOMO repo (`snap-research/locomo`); confirm its
schema and category mapping there. Then:

```bash
# Quick run: 1 conversation, local memory + heuristic scoring (no API cost)
uv run python scripts/eval_locomo.py --dataset path/to/locomo10.json

# Phase 1 raw-turn retrieval baseline; skips answer generation and judge scoring
uv run python scripts/eval_locomo.py \
  --dataset path/to/locomo10.json \
  --retrieve-from turns \
  --out eval_results/locomo_turns_phase1_baseline.jsonl

# Full setup: local memory backend, DeepSeek as a stronger answerer + LLM judge
LLM_PROVIDER=ollama uv run python scripts/eval_locomo.py \
  --dataset path/to/locomo10.json \
  --answerer-provider deepseek --answerer-model deepseek-v4-pro \
  --judge deepseek --judge-model deepseek-v4-pro
```

The memory backend (extraction + reconciliation) follows `LLM_PROVIDER`/`LLM_MODEL`; the answerer
and judge are chosen independently via flags, so you can isolate memory quality from the answerer
LLM. Existing completed session databases can be reused for retrieval experiments because
re-ingestion skips extraction. Start with `--max-conversations 1`; a fresh full LOCOMO conversation
is hundreds of turns and ingests slowly.

### Conv-26 Phase 4 Result

The Phase 4 run reused the completed Phase 3 database for `conv-26`, applied migration 4, and
evaluated all 199 questions at `k=10`. Retrieval used hybrid memories plus raw turns; answers used
`deepseek-v4-pro`, with no separate LLM judge.

| Metric | Prior Phase 3 artifact | Phase 4 | Change |
| --- | ---: | ---: | ---: |
| Evidence recall | 0.516 | **0.700** | **+0.183** |
| Evidence hit rate | 0.558 | **0.736** | **+0.178** |
| Token-overlap recall proxy | 0.349 | **0.472** | **+0.123** |

Phase 4 evidence recall by category was 0.787 adversarial, 0.260 multi-hop, 0.318 open-domain,
0.786 single-hop, and 0.919 temporal. The reported answer accuracy was 0.231 using the harness's
token-overlap heuristic; it is not an LLM-judged accuracy result.

`LongMemEval` is the more rigorous follow-up benchmark (its knowledge-update and temporal-reasoning
categories directly exercise the reconciler); this harness is the skeleton for that run.

## Project Layout

```text
src/brain/
  config.py          # pydantic-settings configuration
  embeddings.py      # Embedder interface, sentence-transformer embedder, fake embedder
  extract.py         # Layer 2 fact extraction prompt/schema and Extractor
  llm.py             # LLMClient, OllamaLLMClient, DeepSeekLLMClient, FakeLLMClient
  memory.py          # MemoryService facade and build_memory factory
  mcp_server.py      # Layer 4 MCP stdio server
  models.py          # shared Pydantic models and Reconciler contract
  reconcile.py       # AlwaysAddReconciler and LLMReconciler
  retrieval.py       # filters, RRF fusion, overfetch settings, optional reranker
  store/
    base.py          # MemoryStore interface
    migrations/      # ordered SQLite schema migrations
    sqlite.py        # SQLiteMemoryStore

scripts/
  demo_layer1.py
  demo_layer2.py
  demo_layer3.py
  eval_locomo.py

tests/
  test_store.py      # memory store tests
  test_ingest.py     # raw session/turn ingestion and retrieval tests
  test_migrations.py # schema migration tests
  test_retrieval.py  # hybrid retrieval, filters, FTS sync, fusion, reranker tests
  test_extract.py    # Layer 2 tests
  test_reconcile.py  # Layer 3 tests
  test_mcp.py        # Layer 4 MCP transport tests
  test_deepseek_llm.py
  test_eval_locomo.py
  fixtures/          # recorded conversations and LLM responses
```

## Layer Completion Log

### Layer 1: Storage + Retrieval

Implemented:

- SQLite + sqlite-vec schema with scoped rows.
- Versioned schema migrations.
- Append-only sessions and raw turns preserving source IDs, speakers, and timestamps.
- Idempotent, retry-safe session ingestion with atomic memory completion.
- FTS5/BM25 raw-turn retrieval.
- FTS5/BM25 memory retrieval synchronized on add, update, and delete.
- RRF fusion over vector and BM25 memory rankings.
- Subject filters with pre-filter overfetch for top-k correctness.
- Unified memory + raw-turn evidence retrieval.
- Optional local cross-encoder reranking.
- Async `MemoryStore` interface.
- `SQLiteMemoryStore` with memory CRUD plus session ingestion, turn lookup, and turn search.
- `SentenceTransformerEmbedder` and deterministic `FakeEmbedder`.
- Layer 1 demo and store tests.

Verification:

```bash
uv run python scripts/demo_layer1.py
uv run pytest tests/test_store.py tests/test_ingest.py tests/test_migrations.py \
  -k "not live_embedder" -v
```

### Layer 2: Extraction + Naive Add

Implemented:

- `LLMClient`, `OllamaLLMClient`, and `FakeLLMClient`.
- `Extractor` with the pinned extraction prompt and JSON schema.
- `FactCandidate`, `MemoryAction`, `MemoryActionKind`, and `Reconciler` contracts.
- `AlwaysAddReconciler`.
- `MemoryService.add` extraction/write loop with ADD, UPDATE, DELETE, and NOOP branches present.
- Pass-through `MemoryService.search`, `get`, and `forget`.
- `build_memory()` factory introduced for the local embedder, SQLite store, and Ollama LLM.
- Layer 2 fixtures, tests, and demo.

Verification:

```bash
uv run pytest tests/test_extract.py -k "not ollama" -v
uv run pytest tests/test_extract.py -k "ollama" -v -s
uv run python scripts/demo_layer2.py --mode fixture
uv run python scripts/demo_layer2.py --mode live
```

### Layer 3: Reconciliation

Implemented:

- `LLMReconciler` with the pinned ADD/UPDATE/DELETE/NOOP decision schema and prompt.
- `target_index` to `Memory.id` resolution with invalid targets falling back to ADD.
- Candidate filtering at cosine-similarity score `0.3` before reconciliation.
- `build_memory()` now wires `LLMReconciler` as the active reconciler.
- UPDATE dispatch returns the updated memory from `MemoryService.add`.
- Recorded pizza-to-sushi fixture, deterministic unit/integration tests, live smoke test, and demo.

Verification:

```bash
uv run pytest tests/test_reconcile.py -v -m "not live"
uv run pytest tests/test_reconcile.py -m live -v -s
uv run python scripts/demo_layer3.py
uv run python scripts/demo_layer3.py --mode live
```

### Layer 4: MCP Server

Implemented:

- `mcp==1.27.2` dependency from the official MCP Python SDK.
- `FastMCP("brain-memory")` stdio server.
- Module-level `build_memory()` construction shared by all tools.
- `remember`, `recall`, `recall_evidence`, and `forget` tools with flat scope arguments.
- Optional subject filters on `recall` and `recall_evidence`.
- In-process MCP tests covering remember/recall persistence across separate calls and forget.

Verification:

```bash
uv run pytest tests/test_mcp.py -v
uv run pytest
uv run python -m brain.mcp_server
```

## Notes

- `ollama` is imported only in `src/brain/llm.py`.
- `AlwaysAddReconciler` intentionally allows duplicate memories; `LLMReconciler` decides whether to
  add, update, delete, or skip.
- `content_hash` exists for future policy, but it is not a uniqueness constraint.
- `id` and `target_id` are strings across store/facade boundaries.
- `Scope` objects should be passed to store and service calls, not loose dictionaries.
- MCP scope arguments are flat on the wire; handlers construct `Scope` internally.
- The MCP server is stdio-only; no auth, HTTP, or SSE transport is configured.
