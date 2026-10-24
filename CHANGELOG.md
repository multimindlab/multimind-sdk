# Changelog

All notable changes to MultiMind SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-12

First stable release of the supported surface: providers, the PII guard,
`multimind serve`, one Swagger API (`multimind backend` and the Docker image),
GDPR and HIPAA checks that do not invent a pass, FAISS and Chroma RAG, and
function-calling agents. Minimum Python is 3.10.

### Fixed
- Workflow executor no longer substitutes echo models for OpenAI, Claude, Gemini, or Ollama. A real `model_registry` is required.
- EU AI Act checks return `unverifiable` when no evidence is stored.
- Hyperparameter search runs Optuna or raises. It no longer returns a hardcoded best parameter.
- Document structure extraction parses the model JSON or raises, instead of returning an empty document.
- `POST /v1/sessions/{id}/messages` returns the assistant reply.
- Compliance monitor uses implemented checkers. The example route, which imported a package that is not in the wheel, is removed.
- Compliance monitor no longer calls unimplemented advanced classes. History and alerts still read the in-memory shard, which is empty until that process records a check.
- The gateway API exposes the SDK providers (Gemini, Groq, Mistral, DeepSeek, OpenRouter, Together, xAI, Perplexity, Fireworks, Cerebras, Claude), not only OpenAI, Anthropic, Ollama, and HuggingFace.
- FAISS search honors `filter_criteria`, including lookup by id.
- Unimplemented vector backends are off the default load path.
- Package version is `multimind.__version__` everywhere.
- Guard proxy appends to a cost log when `costs_log` is set, and `POST /v1/guardrails/reload` applies a guardrails file without a restart.
- `multimind backend` includes the compliance routes on FastAPI 0.130+, which stores an included router lazily; the route copier now descends into it.
- `multimind compliance run-compliance` no longer exits 1 with `'recommendations'` after printing results; recommendations are read from the top level, where the API returns them.
- `multimind --version` prints the package version.
- Compliance dashboard and alerts (`/v1/compliance/dashboard`, `/v1/compliance/alerts`) now reflect every `POST /v1/compliance/monitor` recorded in the server process, with `unverifiable_checks` reported separately and no invented privacy, fairness, or transparency scores. Previously a fresh shard was built per request, so both were always empty.
- `multimind compliance run-example` says whether the `examples` package or `torch` is the missing import.
- Backend API parity with the CLI: `GET /api/costs?period`, `GET /api/inventory?include_env`, `GET /api/evidence-report?period&organization&project&costs_log`, `POST /v1/sessions/{id}/save`, `POST /v1/sessions/load`.
- Governance dashboard gained Scan text, Evidence report, and Compliance views and a "Reload proxy" button after saving guardrails. The guard proxy reads `MULTIMIND_CORS_ORIGINS` so that button works cross-origin.
- Docker image published to `ghcr.io/multimindlab/multimind-sdk` (`linux/amd64` and `linux/arm64`) on every merge to `develop` (`:develop`, `:sha-<short>`) and every release (`:1.0.0`, `:1.0`, `:latest`), after a smoke test of `/health`, the OpenAPI document, and the CLI. `docker compose` now uses the published image. The image installs CPU-only torch, which takes it from 6.5 GB to 1.8 GB with no functional change. The `supported-surface` CI job uses path-based coverage so pydantic 2.14 no longer breaks collection.
- Feature-status audit (2026-10-13), every row verified against code and tests. Fixed: 50 vector store backends could not be constructed (missing `initialize()`), and the unified `VectorStore` wrapper never worked (wrong config attribute, positional config); `SummaryBufferMemory` could not be constructed; `SummaryMemory.get_messages` returned empty strings; `HybridMemory.get_messages` raised `KeyError` with `BufferMemory` children; `MultiMind(memory=...)` raised on every `chat()`; `AdapterTuner`/`PTuner` raised `NameError`; non-streaming Ollama calls failed on NDJSON and sampling options were ignored; four example scripts never imported the SDK or crashed. Added construction tests for all 60 backends, an in-process Chroma round-trip, and tests for Ollama, the OpenAI embedder, differential privacy, adaptive privacy, agent registry, and `MultiMind` with memory. The feature-status page now marks Stable only where a default-suite test exists, lists 6 local, 43 client-backed, and 11 placeholder vector stores, and relabels the "ReAct toolchain" as a sequential reasoning chain.
- Feature chain (2026-10-13): every SDK component is reachable component → backend route → dashboard view → docs, with a test per link. New `multimind/gateway/features_api.py` adds 13 route groups under `/v1` (hallucination evaluation, session switch, context transfer, agents, memory, vector store inventory, synthetic Q/A and fine-tuning capabilities, conversion formats, regulatory check, differential privacy, MCP tools, integrations, tracing) and `GET /v1/features`. New `multimind/config_store.py` with `GET/PUT /api/config` and `POST /api/config/test-provider` manages every setting except API keys; the store wins over env without a restart. `multimind backend` serves the dashboard at `/`, so one process is the playground, and its `/config.js` never embeds a key. The dashboard grew to 26 views with an in-browser API key field. `scripts/check_feature_chain.py` runs in CI.
- Docs: new `tracing` page, full `convert` and `context-transfer` references, eleven feature-status rows, a feature parity matrix under maintainers, and a CI check (`scripts/check_docs_parity.py`) that fails when a CLI command or backend route is not mentioned in `docs/`.
- Docs: every page under `docs/` was verified against the code (memory, architecture, gateway, RAG, and compliance pages rewritten where they described classes or endpoints that do not exist), 26 stale redirect stubs and the Docusaurus site in `multimind-docs/` were removed, and `docs/` is now a Mintlify site (`docs/docs.json`) hosted at docs.multimind.dev. `INSTALLATION.md` and `FEATURES.md` moved to `docs/installation.md` and `docs/feature-status.md`.
- CI holds the PII guard and the guard proxy at 70% line coverage. Providers, agents, and the CLI are held at 40% (measured at 46%; MoE scaffolds and the conversion CLI are still thin). Nightly `requires_api_key` tests run when provider secrets are configured.

The notes below were collected for an unreleased 0.3.0 and ship in 1.0.0.

### Added
- Guard proxy: `multimind serve` — an OpenAI-compatible reverse proxy that
  adds PII guarding, budgets, and audit logging to any existing app
  (see `docs/guard-proxy.md`)
- AI inventory and chargeback: `multimind audit` CLI (see `docs/ai-inventory.md`)
- `multimind scan-text` CLI for scanning text for PII/sensitive data
- Compliance MCP server (`python -m multimind.mcp_server`) for Claude
  Desktop / Claude Code and other MCP clients (see `docs/mcp-server.md`)
- Framework adapters for LangChain, LlamaIndex, CrewAI, and the OpenAI SDK
  (see `docs/integrations.md`)
- New extras: `mcp`, `langchain`, `llamaindex`, `crewai`
- `pyproject.toml` with modular extras: `rag`, `vector-stores`, `agents`,
  `memory`, `documents`, `finetune`, `finetune-gpu`, `compliance`, `gateway`,
  `dev`, `all` (#41)
- PEP 562 lazy imports — `from multimind import OpenAIModel` no longer pulls
  in `torch`, `transformers`, `chromadb`, etc. on minimal installs
- Helpful `ImportError` messages from subpackages naming the right extras
  group to install (e.g. `pip install multimind-sdk[rag]`)
- `multimind/_lazy.py` helper module (`import_optional`, `lazy_attr`)
- `tests/conftest.py` with `mock_openai_key` / `mock_anthropic_key` fixtures,
  `requires_torch` / `requires_faiss` / `requires_plotly` etc. skip
  decorators, and an auto-skip collection hook for tests marked
  `@pytest.mark.requires_api_key` when no keys are in the environment
- `plotly`, `dash`, and `pandas` added to `[compliance]` extra (powers the
  compliance visualization dashboards)
- `optuna` added to `[finetune]` extra (was eagerly imported but undeclared)
- Multi-job CI workflow (`.github/workflows/ci.yml`): `lint`, `test-core`
  (Python 3.10 / 3.11 / 3.12 / 3.13 matrix), `test-rag`,
  `test-compliance`, `test-finetune`, `test-gateway`, `test-full`
  (coverage + 95% pass-rate gate)
- `CHANGELOG.md` (this file)
- Re-exports for `SlackIntegrationHandler` and `JiraIntegrationHandler` on
  `multimind.integrations`

### Changed
- Minimum Python version is **3.10** (1.0.0). The modernization pass had used 3.9.
- Version is **1.0.0**
- Complete README rewrite (660 → ~180 lines): single Quick Start,
  honest feature table, working code examples verified against the actual
  API surface
- Comprehensive `.gitignore` (build artifacts, IDE files, coverage,
  database files, OS metadata, jupyter checkpoints, ruff/mypy caches)
- `multimind/client/model_client.py`: `LSTMModelClient`, `RNNModelClient`,
  `GRUModelClient` now call `torch.load(weights_only=False)` for
  PyTorch ≥ 2.6 compatibility (trusted-load is appropriate since users
  load their own training checkpoints)
- Narrowed enforced Ruff rule set to `E, F, W, I` in `pyproject.toml`;
  `UP`/`N` rules disabled until Pydantic v1 `@validator` methods can be
  migrated safely (see `pyproject.toml` for rationale)
- Tooling configuration consolidated into `pyproject.toml`
  (`[tool.pytest.ini_options]`, `[tool.ruff]`, `[tool.black]`,
  `[tool.mypy]`); `pytest.ini` and `.flake8` deleted
- Moved root-level doc files to `docs/`:
  - `README-llm.md` → `docs/llm-integration.md`
  - `llm-crawler-guide.md` → `docs/llm-crawler-guide.md`
  - `llm.txt` → `docs/llm.txt`
  - `multimind-sdk-metadata.json` → `docs/multimind-sdk-metadata.json`

### Fixed
- Circular import in `multimind.compliance` — removed redundant
  `run_compliance` re-export from `multimind.compliance.__init__`
  (the function remains available at `multimind.cli.compliance.run_compliance`)
- Test discovery: `pytest tests/` now works regardless of invocation
  (`pythonpath = ["."]` added to pytest config); previously some tests
  required `PYTHONPATH=$PWD` as a CI workaround
- `multimind/vector_store/typesense.py` — nested same-quote f-string
  syntax error that prevented the module from loading on Python ≥ 3.11
- `examples/mcp/__init__.py` — removed imports of modules that no longer
  exist (`.ci_cd_workflow`, `.code_review_workflow`, etc.); package init
  is now lightweight, callers import specific examples directly
- `multimind/fine_tuning/qlora_trainer.py` — removed unused
  `import bitsandbytes as bnb` so the module loads on macOS/ARM and
  any host without a GPU
- `multimind/ensemble/advanced.py` — replaced 4 bare `except:` with
  explicit `except Exception:` (preserves keyboard-interrupt semantics)
- `tests/test_import.py` — three tests used `return True` instead of
  `assert`; pytest 9 will error on this pattern
- `tests/test_retrieval.py` — two async retrieval tests were missing
  `await`, so the coroutines were silently dropped and the tests were
  trivially "passing" without exercising any code
- `tests/test_model_client.py` — `DummyModel` moved to module scope so
  `torch.save` / `torch.load` round-trips work; shape corrected to
  `[batch, seq_len, vocab]`; three previously-skipped tests now run
- `tests/test_document_loader.py` — replaced `Mock(spec=BaseLLM)` with
  `AsyncMock(spec=BaseLLM)` so the language-detect path
  (`response.strip().lower()`) gets a real string
- 6,394 lint issues auto-fixed via `ruff check --fix` + `black`
  (whitespace, import sorting, end-of-file newlines, redundant
  `open()` modes, etc.) across 347 files in `multimind/`

### Removed
- `setup.py` was kept as a no-op shim in `92294804` for backward
  compatibility; further removal will follow once tooling has had time
  to settle on `pyproject.toml`
- `pytest.ini` (replaced by `[tool.pytest.ini_options]`)
- `.flake8` (replaced by Ruff configuration)
- `requirements-base.txt`, `requirements-compliance.txt`,
  `requirements-dev.txt` (replaced by `[project.optional-dependencies]`)
- `TEST_FIXES_COMPLETE.md` (internal tracking — no longer needed in
  the repo root)
- `[local]` install extra (the `ollama` PyPI package was never used;
  `multimind.models.ollama` talks to Ollama directly over HTTP via
  `aiohttp`)

### Security
- Zero-knowledge proof helpers raise `NotImplementedError`. They do not
  report a successful proof.

## [0.2.2] - 2025-08-18

### Fixed
- Model router error (#51)
- Streaming, memory, SQLAlchemy cleanup, and reliability issues across
  the SDK

### Changed
- Refactored Unified API for stable Mixture-of-Experts execution with
  multi-modal routing and dynamic expert support

## [0.2.1] - 2025-07

### Added
- Initial public release: multi-model chat, RAG, agents, basic
  compliance framework
