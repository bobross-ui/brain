# Brain

Local-first agentic memory layer for agents. The project stores scoped user memories in
SQLite with vector search, extracts atomic facts from conversations, and is being built in
layers toward an MCP stdio server.

## Status

| Layer | Status | What works now |
| --- | --- | --- |
| Layer 1: Storage + Retrieval | Complete | SQLite + sqlite-vec store, local embedder, scoped add/search/get/delete/update, Layer 1 demo |
| Layer 2: Extraction + Naive Add | Complete | Ollama-backed fact extraction, fake LLM fixture path, always-add reconciler, MemoryService facade, Layer 2 demo |
| Layer 3: Reconciliation | Not started | Planned ADD/UPDATE/DELETE/NOOP LLM reconciler |
| Layer 4: MCP Server | Not started | Planned MCP tools over stdio |

Current behavior: Layer 2 extracts facts from conversation messages and stores each fact as a
new memory. Duplicates are allowed. There is no reconciliation, deduplication, MCP server, auth,
or HTTP transport yet.

## Architecture

The core seams are:

- `Embedder`: converts text into vectors. Current implementation: `SentenceTransformerEmbedder`.
- `MemoryStore`: async storage interface. Current implementation: `SQLiteMemoryStore`.
- `LLMClient`: async JSON chat interface. Current implementation: `OllamaLLMClient`.
- `Reconciler`: decides what to do with extracted facts. Current implementation: `AlwaysAddReconciler`.
- `MemoryService`: facade used by later layers: `add`, `search`, `get`, and `forget`.

Memory scope is always explicit through `Scope(user_id, agent_id, namespace)`. Layer 1/2 search and
read paths filter by `user_id` and `namespace`.

## Requirements

- Python 3.11+
- `uv`
- macOS/Linux with local SQLite extension support
- Ollama for live Layer 2 extraction

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
LLM_MODEL=qwen2.5:3b-instruct
LLM_PROVIDER=ollama
```

No API keys are required.

## Ollama Setup

Layer 2 live extraction uses Ollama with `qwen2.5:3b-instruct`.

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

## Tests

Run deterministic Layer 2 tests:

```bash
uv run pytest tests/test_extract.py -k "not ollama" -v
```

Run non-live tests for completed layers:

```bash
uv run pytest -k "not slow and not ollama" -v
```

Run live Layer 2 Ollama smoke tests:

```bash
ollama serve
uv run pytest tests/test_extract.py -k "ollama" -v -s
```

Run the live local embedder smoke test. This may download the sentence-transformers model on first
run:

```bash
uv run pytest tests/test_store.py -k "live_embedder" -v -s
```

## Project Layout

```text
src/brain/
  config.py          # pydantic-settings configuration
  embeddings.py      # Embedder interface, sentence-transformer embedder, fake embedder
  extract.py         # Layer 2 fact extraction prompt/schema and Extractor
  llm.py             # LLMClient, OllamaLLMClient, FakeLLMClient
  memory.py          # MemoryService facade and build_memory factory
  models.py          # shared Pydantic models and Reconciler contract
  reconcile.py       # AlwaysAddReconciler
  store/
    base.py          # MemoryStore interface
    schema.sql       # SQLite/sqlite-vec schema
    sqlite.py        # SQLiteMemoryStore

scripts/
  demo_layer1.py
  demo_layer2.py

tests/
  test_store.py      # Layer 1 tests
  test_extract.py    # Layer 2 tests
  fixtures/          # recorded conversations and LLM responses
```

## Layer Completion Log

### Layer 1: Storage + Retrieval

Implemented:

- SQLite + sqlite-vec schema with scoped rows.
- Async `MemoryStore` interface.
- `SQLiteMemoryStore` with `add`, `search`, `get`, `delete`, and `update`.
- `SentenceTransformerEmbedder` and deterministic `FakeEmbedder`.
- Layer 1 demo and store tests.

Verification:

```bash
uv run python scripts/demo_layer1.py
uv run pytest tests/test_store.py -k "not live_embedder" -v
```

### Layer 2: Extraction + Naive Add

Implemented:

- `LLMClient`, `OllamaLLMClient`, and `FakeLLMClient`.
- `Extractor` with the pinned extraction prompt and JSON schema.
- `FactCandidate`, `MemoryAction`, `MemoryActionKind`, and `Reconciler` contracts.
- `AlwaysAddReconciler`.
- `MemoryService.add` extraction/write loop with ADD, UPDATE, DELETE, and NOOP branches present.
- Pass-through `MemoryService.search`, `get`, and `forget`.
- `build_memory()` wiring local embedder, SQLite store, Ollama LLM, and always-add reconciler.
- Layer 2 fixtures, tests, and demo.

Verification:

```bash
uv run pytest tests/test_extract.py -k "not ollama" -v
uv run pytest tests/test_extract.py -k "ollama" -v -s
uv run python scripts/demo_layer2.py --mode fixture
uv run python scripts/demo_layer2.py --mode live
```

## Notes

- `ollama` is imported only in `src/brain/llm.py`.
- Layer 2 intentionally allows duplicate memories.
- `content_hash` exists for future policy, but it is not a uniqueness constraint.
- `id` and `target_id` are strings across store/facade boundaries.
- `Scope` objects should be passed to store and service calls, not loose dictionaries.
