# Changelog

All notable changes to MultiMind SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Targets `0.3.0` — first release after the packaging modernization, lazy-import,
and test-stabilization passes.

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
  (Python 3.9 / 3.10 / 3.11 / 3.12 / 3.13 matrix), `test-rag`,
  `test-compliance`, `test-finetune`, `test-gateway`, `test-full`
  (coverage + 95% pass-rate gate)
- `CHANGELOG.md` (this file)
- Re-exports for `SlackIntegrationHandler` and `JiraIntegrationHandler` on
  `multimind.integrations`

### Changed
- Minimum Python version raised to **3.9**
- Version bumped to **0.3.0**
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
- `cryptography.zkp` ZeroKnowledgeProof falls back to a clearly-named
  `_DummyZKP` with a `UserWarning` when the optional dependency is not
  installed. **This is not a production-grade ZKP** — install the real
  dependency or use a different compliance pathway for any guarantee
  that needs to survive audit.

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
