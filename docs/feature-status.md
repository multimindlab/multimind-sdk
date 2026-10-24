---
title: "MultiMind SDK - Feature Status"
description: "Verified status of every MultiMind SDK feature: what is tested and working, what exists but is untested, and what is a placeholder."
---

This page is the source of truth for what the SDK does today. Every row was checked against the code and the test suite on 2026-10-13, and the badges follow one rule:

- ✅ **Stable**: works, and a test in the default suite proves it
- 🚧 **Beta**: real code that constructs and runs, but with a stated limitation or without a test in the default suite
- 📋 **Planned**: not implemented, or a placeholder that raises `NotImplementedError` rather than pretending to work

A row never says a feature works unless something runs it. Where the proof needs a live service or a GPU, the row says so and stays Beta.

---

## 🧠 Core AI Model Management

### Model Integrations
- ✅ **OpenAI** (`OpenAIModel`) - chat, generation, embeddings, vision input
  - Example: `examples/api/model_wrapper.py`; tests: `tests/test_new_providers.py`, `tests/test_vision.py`
- ✅ **Claude** (`ClaudeModel`) - chat, generation, vision input
  - Example: `examples/api/model_wrapper.py`; tests: `tests/test_new_providers.py`, `tests/test_vision.py`
- ✅ **Ollama** (`OllamaModel`) - local models over HTTP. Two bugs fixed on 2026-10-13: non-streaming calls never sent `stream: false`, so every `generate()` and `chat()` against a real server failed on the NDJSON response; and `temperature`/`max_tokens` were sent at the top level, where Ollama ignores them, instead of under `options` as `temperature`/`num_predict`
  - Example: `examples/cli/chat_ollama_cli.py`; tests: `tests/test_ollama_model.py`
- ✅ **Mistral** - `MistralAIModel` (Mistral API, OpenAI-compatible) and `MistralModel` (the Ollama model `mistral`)
- ✅ **Gemini**, **Groq**, **DeepSeek** (`GeminiModel`, `GroqModel`, `DeepSeekModel`)
- ✅ **OpenRouter** (`OpenRouterModel`) - one key for the models OpenRouter aggregates; `embeddings()` raises `NotImplementedError` by design
- ✅ **Together AI** (`TogetherModel`) - chat and embeddings
- ✅ **xAI** (`XAIModel`) - chat; no embeddings endpoint
- ✅ **Perplexity** (`PerplexityModel`) - Sonar models
- ✅ **Fireworks AI** (`FireworksModel`) - chat and embeddings
- ✅ **Cerebras** (`CerebrasModel`)
- ✅ **13 first-party providers** - the thirteen classes above, each tested with a mocked client in `tests/test_new_providers.py`. `HuggingFaceModel` runs local transformers models and needs the `[finetune]` extra. Model counts per provider depend on the upstream catalog; the SDK ships no model list

### Model Features
- ✅ **Vision/Multimodal Input** - `images=` on `generate`/`chat` for OpenAI-compatible providers (content parts with base64 data URLs or URLs) and Claude (image blocks); providers without a vision endpoint raise `NotImplementedError`
  - Example: `examples/models/vision_demo.py`; tests: `tests/test_vision.py`
- ✅ **Multi-Model Wrapper** (`MultiModelWrapper`) - one interface over several models with per-model performance stats
  - Example: `examples/api/multi_model_wrapper.py`; tests: `tests/examples/cli/test_multi_model_wrapper_cli.py`, `tests/test_basic.py`
- ✅ **Model Routing** (`Router`, `ModelRouter`) - rule-based routing between models
  - Example: `examples/api/ensemble_api.py`; tests: `tests/test_orchestration_router.py`, `tests/examples/api/test_ensemble_api.py`
- ✅ **Model Sessions** (`ModelSession`) - stateful chat over any model with history, a rolling summary past the token budget, and `switch(model, strategy=...)` with `full`, `summary`, or `adapted` transfer
  - Tests: `tests/test_model_session.py`
- ✅ **Context Transfer** (`multimind.context_transfer`, `multimind context-transfer`) - reformats a saved conversation from one provider into a prompt for another with last-N extraction, three summary styles, and provider adapters. Two tests for private formatter methods that no longer exist are skipped, not failing
  - Tests: `tests/test_context_transfer.py`, `tests/test_context_transfer_integration.py`, `tests/test_cli.py`
- 🚧 **Model Conversion** (`ModelConversionManager`, `multimind convert`) - HuggingFace, PyTorch, TensorFlow, ONNX, and Ollama sources to GGUF, Safetensors, TFLite, ORT, and ONNX targets. Needs the `[finetune]` extra. Only the no-torch failure path is tested
  - Tests: `tests/test_cli.py`
- 🚧 **Mixture-of-Experts** (`MoEFactory`, `MoEModel`) - config validation tested without torch; routing and capacity tests need torch and skip in the default suite. Two factories exist (`models/moe.py` and `models/moe/moe_factory.py`)
  - Example: `examples/moe/`; tests: `tests/test_moe_edge_cases.py`
- 🚧 **Model Compression** - dynamic quantization. All tests need torch and skip in the default suite
  - Tests: `tests/test_model_compression_edge_cases.py`
- 📋 **Federated Learning** - `memory/federated.py` and `client/federated_router.py` raise `NotImplementedError`; no training
- 📋 **Model Watermarking** - `ModelWatermarking.watermark_model` and `verify_watermark` raise `NotImplementedError`; only `track_fingerprint` works

---

## 📚 Vector Databases & RAG

### Vector Store Backends

The SDK ships 60 backend modules. On 2026-10-13 every one of them was made constructible: 50 were missing the `initialize()` override the base class requires, so they raised `TypeError` before touching a client. The unified `VectorStore` wrapper was also fixed; it read a config attribute that did not exist and passed the config positionally to backends that take keyword arguments, so it had never worked with any backend.

- Tests: `tests/test_vector_store_construction.py` (every backend constructs; local backends round-trip; the unified wrapper round-trips FAISS and Chroma), `tests/integration/test_vector_stores_live.py` (`pytest -m integration`)

#### ✅ **Local** - tested end to end in the default suite
- ✅ **FAISS** - add, search, delete, persist, load; honors `filter_criteria` including lookup by id
- ✅ **Chroma** - in-process persistent client; add, search, metadata filter, delete, reopen from disk. `persist_directory` in the config is honored
- ✅ **Annoy**, **sklearn**, **SQLiteVSS** - add, search, delete round-trips
- ✅ **LanceDB** - constructs and initializes against the installed library; no round-trip test yet beyond construction

#### 🚧 **Client-backed** - real client code, constructible, needs a live service to prove
Each of these imports its client library lazily, raises a clear `ImportError` naming the pip package when it is missing, and calls the real client in `add_vectors`, `search`, and `delete_vectors`. None is exercised against a live service in CI, so none is Stable.

Pinecone, Qdrant, Weaviate, Milvus, Zilliz, Elasticsearch, OpenSearch, MongoDB Atlas, Atlas, PGVector, PGVectoRS, PGEmbedding, AlibabaCloud OpenSearch, AnalyticDB, AstraDB, Hippo, Hologres, Marqo, Matching Engine, Meilisearch, Momento, MyScale, Neo4j Vector, NucliaDB, Rockset, SingleStoreDB, StarRocks, Supabase, Tair, TencentVectorDB, Tigris, TimescaleVector, Typesense, USearch, Vald, Vectara, Xata, Zep.

Partially implemented, where the unsupported operation raises `NotImplementedError`: Cassandra (no search), Azure AI Search and Azure Cosmos DB (no vector search), LLMRails (no delete), TileDB (no delete).

#### 📋 **Not implemented** - every operation raises `NotImplementedError`
AwaDB, BagelDB, BaiduCloud, Clarifai, ClickHouse, DashVector, Databricks Vector Search, DeepLake, DingoDB, Elastic Vector Search (legacy module), Epsilla. These load only when `MULTIMIND_EXPERIMENTAL_VECTOR_STORES=1` is set.

> Count: 6 local backends tested in the default suite, 43 client-backed backends that construct and have real client code but no CI proof, 11 placeholders.

### RAG Features
- ✅ **Basic RAG Pipeline** (`RAG`, `RAGPipeline`) - add documents, retrieve, generate
  - Example: `examples/rag/example_rag.py`; tests: `tests/examples/rag/`, `tests/test_api_rag.py`
- ✅ **Document Processing** (`TextSplitter`, `document_processing`) - splitting and chunking
  - Example: `examples/rag/example_rag.py`; tests: `tests/test_splitter.py`, `tests/test_advanced_document_processor.py`
- ✅ **Embedding Management** (`EmbeddingGenerator`, `OpenAIEmbedder`) - batching, normalization, dimension handling, stats
  - Tests: `tests/test_embeddings_openai.py`. `HuggingFaceEmbedder` needs torch and is not tested in the default suite
- 🚧 **Advanced RAG** (`Retriever`, `EnhancedRetriever`, `HybridRetriever`) - metadata filtering and fusion. Known bug: `RAG.retrieve(filter_criteria=...)` spreads the dict as keyword arguments into the vector store search and raises; filter client-side until fixed
  - Example: `examples/rag/self_tuning_rag_example.py`; tests: `tests/test_advanced_rag_edge_cases.py`, `tests/test_retrieval.py`
- ✅ **Hybrid RAG with knowledge graph** (`GraphRetriever`, `GraphRAG`) - entity and relation graph built by LLM extraction, ranked by query-entity overlap with one-hop neighbor expansion
  - Tests: `tests/test_graph_retrieval.py`
- 📋 **Quantum-Enhanced Search** - not implemented
- 🚧 **Multi-Modal Document Processing** - table extraction via pdfplumber and pandas; image OCR via pytesseract (`[documents]` extra); no object detection or captioning; LLM-based structure analysis is a placeholder

---

## 🤖 AI Agents

### Agent Framework
- ✅ **Basic Agents** (`Agent`) - native function calling: JSON-schema tool definitions, a bounded `tool_calls` loop (`max_turns`), results fed back to the model; keyword-based tool routing when the model has no tool API
  - Example: `examples/cli/basic_agent.py`; tests: `tests/test_agents_function_calling.py`
- ✅ **Agent Registry** (`AgentRegistry`, exported from `multimind.agents`) - registration, lookup, and fallback
  - Example: `examples/agents/agent_registry_example.py`; tests: `tests/test_agents_registry_and_chain.py`
- 🚧 **Reasoning Chain** (`ReasoningChain`, `agents/react_toolchain.py`) - a sequential pipeline of callables with hooks. It is not a ReAct loop: there is no model-driven thought, action, observation cycle and no tool selection. The example runs offline
  - Example: `examples/agents/react_toolchain_example.py`; tests: `tests/test_agents_registry_and_chain.py`
- ✅ **Multi-Agent Orchestration** (`AgentOrchestrator`) - delegation and spawning with depth, agent-count, and turn bounds, plus an audit hook
  - Tests: `tests/test_agent_orchestration.py`
- ✅ **Self-Evolving Agents** (`SelfEvolvingAgent`) - records task outcomes and injects top-utility few-shot exemplars; no self-modification of code or prompts
  - Tests: `tests/test_self_evolving.py`
- ✅ **Cognitive Scratchpad** - step dependencies with cycle checks and revision history
  - Tests: `tests/test_cognitive_scratchpad.py`
- ✅ **`MultiMind` facade with memory** - `MultiMind(llm, memory=...)` now records turns through the memory API that exists. Until 2026-10-13 it called a method no memory class implemented, so any `chat()` with a memory raised
  - Tests: `tests/test_multimind_memory.py`

---

## 🧠 Memory Systems

Memory classes that are abstract and cannot be instantiated are listed on the [memory page](/memory), not here.

### ✅ **Stable** Memory Types
- ✅ **Buffer Memory** - conversation buffer with save and load
  - Tests: `tests/test_advanced_features.py`
- ✅ **Summary Memory** - LLM-backed rolling summaries. `get_messages()` returned empty strings until 2026-10-13 because it read the wrong key
  - Tests: `tests/test_advanced_features.py`
- ✅ **Summary Buffer Memory** - recent messages plus a summary of older ones. Could not be constructed at all until 2026-10-13: it passed an argument its parent does not accept
  - Tests: `tests/test_memory.py`
- ✅ **Agent Memory** (`AgentMemory`) - task, response, and state tracking
  - Example: `examples/memory/basic_usage.py`; tests: `tests/test_advanced_features.py`, `tests/test_multimind_memory.py`
- ✅ **Vector Store Memory** - add, get, search, clear, save, and load against FAISS with correct similarity ordering
  - Tests: `tests/test_memory_beta.py`
- ✅ **Episodic Memory** - spatial, temporal, and emotional indices, episode chaining, persistence
  - Tests: `tests/test_memory_beta.py`
- ✅ **Semantic Memory** - concept extraction, relationship inference, persistence
  - Tests: `tests/test_memory_beta.py`
- ✅ **Procedural Memory** - procedure extraction, execution recording, adaptation, persistence
  - Tests: `tests/test_memory_beta.py`
- ✅ **Hybrid Memory** - routes across child memories with stats and persistence. Until 2026-10-13 `get_messages()` raised `KeyError` whenever a child such as `BufferMemory` emitted messages without timestamps, which was the documented configuration
  - Example: `examples/memory/advanced_memory_manager.py`; tests: `tests/test_memory_beta.py`, `tests/test_memory_hybrid_buffer.py`
- ✅ **Planning Memory** - plan-step recording with success-weighted next-step suggestion
  - Tests: `tests/test_planning_memory.py`

### 🚧 **Beta** Memory Types
- 🚧 **Consensus Memory** - in-process replication with majority-vote reads; networked RAFT raises `NotImplementedError`
- 🚧 **Declarative Memory** - facts with temporal and causal relations and a knowledge-graph update path; analyses need a JSON-capable LLM
- 🚧 **Implicit Memory** - skill tracking with a prerequisite graph
- 🚧 **Reinforcement Memory** - tabular Q-learning for keep, evict, and compress decisions
- 🚧 **Generative Memory** - regeneration bookkeeping and drift tracking with a caller-supplied generator
- 🚧 **Active Learning Memory** - utility-weighted retrieval and low-utility eviction
  - Tests for all six: `tests/test_memory_advanced.py`, `tests/test_declarative_memory.py`

### 📋 **Planned** Memory Types
- 📋 **Quantum Memory** - `QRAM`, `QAM`, `TopologicalMemory`, and `QuantumClassicalHybridMemory` are abstract scaffolds that cannot be instantiated. `examples/memory/quantum_memory.py` demonstrates that honestly by printing each `TypeError`

---

## 🔄 Fine-Tuning & Model Training

Everything in this section needs torch, which the default test suite does not install. Nothing here is Stable because nothing here is proven in CI.

- 🚧 **Basic LoRA** (`LoRATrainer`) - a peft and `Trainer` wrapper; no tests
  - Example: `examples/fine_tuning/`
- 🚧 **Adapter Training** (`AdapterTuner`, `PTuner`) - constructors raised `NameError` on a missing import until 2026-10-13; no runtime tests
  - Tests: `tests/test_adapter_tuning_imports.py` (import check only)
- 🚧 **Non-Transformer Models** - `MambaLLM` and `RWKVLLM` load through HuggingFace; Hyena, S4 variants, RetNet, H3, MLP-only, and diffusion-text are scaffolds that raise `NotImplementedError`
  - Example: `examples/non_transformer/`
- 🚧 **QLoRA** (`QLoRALLM`, `QLoraTuner`) - 4-bit NF4 plus LoRA through transformers, peft, and bitsandbytes (`[finetune-gpu]`); raises a clear error when those are absent. `CompacterLLM` is a scaffold
  - Tests: `tests/test_llm.py`, `tests/test_llm_wrappers.py` (no-dependency paths only)
- 📋 **HyperLoRA** (`HyperLoRATuner`) - builds hypernetworks that are never wired into training; `train()` runs plain LoRA
- ✅ **Synthetic Q/A for fine-tuning** (`SyntheticQAGenerator`) - Q/A pairs from document chunks via strictly parsed LLM JSON, deduplicated, quality-filtered, exported as Alpaca-style JSONL; pure Python
  - Tests: `tests/test_synthetic_data.py`
- 🚧 **Advanced Optimization** (`BayesianOptimizer`, `KnowledgeDistillation`) - code exists, needs torch and scikit-learn, no tests

---

## 🔐 Compliance & Security

### ✅ **Stable**
- ✅ **PII Guard** (`ComplianceGuard`, `PIIDetector`, `guard(model)`) - detection and redaction (mask, hash, remove) on prompts and responses, block lists, and a JSONL audit trail
  - Tests: `tests/test_compliance_guard.py`
- ✅ **Healthcare Compliance** - PHI tracking with audit-logged access, breach modeling, and state-based HIPAA and HITECH rule checks; requirements needing external systems report `unverifiable`, never a fabricated pass
  - Example: `examples/compliance/healthcare/`; tests: `tests/compliance/test_healthcare.py`
- ✅ **GDPR Runtime Enforcement** (`GDPRPolicy`, `gdpr_enforce`) - evaluates a processing request against lawful basis, purpose limitation, and retention and returns `allowed`, `denied`, or `not_configured` with reasons; wires the policy into the runtime guard
  - Tests: `tests/compliance/test_gdpr_enforcement.py`
- ✅ **Framework Governance Adapters** - PII guard, budget, cost, and audit wrappers for LangChain, LlamaIndex, CrewAI, AutoGen, Haystack, and raw OpenAI clients; lazy imports with install hints. Tested with stubs; not exercised against the installed frameworks in CI
  - Docs: [integrations](/integrations); tests: `tests/test_framework_adapters.py`, `tests/test_framework_adapter_autogen.py`, `tests/test_framework_adapter_haystack.py`
- ✅ **Guard Proxy** (`multimind serve`) - OpenAI-compatible proxy with redaction, block lists, budgets, audit and cost logs, output scanning, guardrails hot reload, and `MULTIMIND_CORS_ORIGINS`
  - Docs: [guard proxy](/guard-proxy); tests: `tests/test_guard_proxy.py`
- ✅ **Hallucination Detection** (`HallucinationDetector`, `GuardedModel`, `HallucinationError`, `detect_hallucinations`) - sentence-level grounding against supplied sources, claim verification with a judge model, and annotate, raise, or retry policies
  - Tests: `tests/test_hallucination.py`
- ✅ **Evidence Reports** (`build_evidence_report`, `multimind compliance report-evidence`) - Markdown or HTML from the audit log, cost log, and inventory scan, mapped to EU AI Act, SOC 2, and HIPAA control themes; absent inputs are stated as absent
  - Tests: `tests/test_compliance_reporting.py`
- ✅ **Governance Dashboard and Playground** (`multimind dashboard`, or `/` on `multimind backend`) - 26 views: governance (audit, costs, inventory, guardrails editor, scan text, evidence report, compliance), a playground view for every feature on this page (chat, compare, sessions, models, RAG, hallucination, agents, memory, vector stores, fine-tuning, conversion, regulatory watch, differential privacy, MCP tools, integrations, tracing, context transfer), and settings for every option except API keys
  - Docs: [dashboard](/dashboard); tests: `tests/test_dashboard.py`, `tests/test_dashboard_views.py`
- ✅ **MCP Server** (`python -m multimind.mcp_server`, `[mcp]`) - `scan_text`, `redact_text`, `check_grounding`, `check_policy`, and `audit_log` tools; findings carry spans, never raw PII
  - Docs: [MCP server](/mcp-server); tests: `tests/test_mcp_server.py`
- ✅ **Regulatory Change Detection** (`RegulatoryWatcher`) - fetches source pages, hashes content, and reports `ChangeEvent`s with a unified diff; `check_offline()` for dry runs. Distinct from the legacy `RegulatoryChangeDetector` stub below
  - Example: `examples/compliance/regulatory_watch_demo.py`; tests: `tests/test_regulatory_watch.py`
- ✅ **Differential Privacy** (`DPMechanism`, `PrivacyBudget`, `BudgetExhaustedError`) - Laplace noise with sequential epsilon accounting; advanced composition is not implemented
  - Tests: `tests/test_differential_privacy.py`
- ✅ **Compliance monitor API** (`POST /v1/compliance/monitor`) - PII scan of supplied text and GDPR purpose checks; anything without runtime evidence is `unverifiable`. Results are recorded per server process and feed `/v1/compliance/dashboard` and `/alerts`
  - Tests: `tests/test_platform.py`

### 🚧 **Beta**
- 🚧 **Adaptive Privacy** (`AdaptivePrivacy`) - adjusts epsilon from compliance-score and loss feedback within configured bounds. Its starting epsilon disagrees with the encryption stub's until the first adaptation
  - Tests: `tests/test_differential_privacy.py`
- 🚧 **Data Governance** (`GovernanceConfig`, `ComplianceMetadata`, `DataProtectionManager`) - configuration models and a data-protection manager; there is no governance rule engine

### 📋 **Planned** - fail-closed stubs
- 📋 **Zero-Knowledge Proofs** - `ZeroKnowledgeProof.prove` and `verify` raise `NotImplementedError`; no ZKP backend
- 📋 **Homomorphic Encryption** - `HomomorphicEncryption.encrypt` raises `NotImplementedError`; nothing is encrypted
- 📋 **Federated Compliance** - `FederatedCompliance.verify_global_compliance` and `ComplianceShard.verify_compliance` raise `NotImplementedError`; the shard's in-memory history and alerts are filled only by the compliance monitor API
- 📋 **Self-Healing Compliance** - `check_and_heal` returns the input unchanged and raises when a vulnerability is detected
- 📋 **`RegulatoryChangeDetector`** - legacy stub whose constructor raises `AttributeError`; use `RegulatoryWatcher`
- 📋 **Quantum-Safe Encryption** - not implemented

---

## 🔄 Workflow & Orchestration
- ✅ **Prompt Chains** (`PromptChain`) - sequential prompt chaining
  - Example: `examples/cli/prompt_chain.py`; tests: `tests/test_basic.py`
- ✅ **Task Runner** (`TaskRunner`) - simple task execution
  - Example: `examples/cli/task_runner.py`; tests: `tests/test_orchestration_router.py`
- 🚧 **MCP workflows** (Model Composition Protocol, `multimind.mcp`) - executor and parser for MultiMind's internal workflow format; requires a real `model_registry`. Not Anthropic's Model Context Protocol, which is the Stable MCP server above
  - Example: `examples/mcp_workflows/`; tests: `tests/examples/cli/test_mcp_workflow.py`, `tests/test_release_fixes.py`
- 🚧 **Pipeline Builder** - basic pipeline construction
  - Example: `examples/pipeline/pipeline_example.py`; tests: `tests/test_pipeline.py`
- 📋 **Visual Workflow Builder**, **Event-Driven Architecture**, **Advanced Workflow Automation** - not implemented

---

## 📊 Monitoring & Analytics
- ✅ **Cost Tracking and Budgets** (`CostTracker`, `Budget`, `track_costs`) - per-call cost records from real token usage where the provider reports it, flagged estimates otherwise, unpriced models recorded at zero and flagged, JSONL persistence, and budget warnings and caps
  - Tests: `tests/test_cost_tracker.py`
- ✅ **Shadow-AI Inventory and Chargeback** (`scan_project`, `CostTracker.chargeback`, `multimind audit`) - static scan of dependencies, API URLs, env-var reads, and hardcoded keys (by location, never by value), plus per-tag chargeback
  - Docs: [AI inventory](/ai-inventory); tests: `tests/test_ai_inventory.py`
- ✅ **Run Tracing** (`RunTracer`, `trace_model`) - runs and nested spans with latency, tokens, and cost to SQLite or a self-hosted platform; hashes and lengths only unless `store_content=True`
  - Docs: [tracing](/tracing); tests: `tests/test_tracing.py`
- 🚧 **Basic Logging** (`TraceLogger`) and **Usage Tracking** (`UsageTracker`) - lightweight helpers
  - Example: `examples/cli/usage_tracking.py`; tests: `tests/test_basic.py`, `tests/examples/cli/test_usage_tracking.py`
- 📋 **Real-time Performance Tracking**, **Anomaly Detection**, **Cost Optimization Engine**, **Predictive Maintenance** - not implemented

---

## 🛠️ CLI & API
- ✅ **CLI** - eight command groups: `audit`, `backend`, `chat`, `compliance`, `config`, `dashboard`, `models`, `serve`, plus `convert` and `context-transfer`, and `--version`
  - Docs: [CLI](/cli); tests: `tests/test_cli.py`
- ✅ **Backend API** (`multimind backend`) - one FastAPI app and one Swagger UI for chat, sessions, compliance, governance, the guard proxy, and RAG, with optional `X-API-Key` auth
  - Docs: [API overview](/api_reference/overview); tests: `tests/test_platform.py`, `tests/test_api_gateway.py`, `tests/test_dashboard.py`
- 🚧 **Compliance alerts** - `POST /v1/compliance/alerts/configure` stores rules in memory; only `enabled: false` is evaluated and there are no notification channels

- ✅ **Feature routes and registry** - every component above has a route under `/v1` and `GET /v1/features` reports each one's availability and routes; settings are managed through `GET/PUT /api/config` with API keys kept in the environment
  - Docs: [gateway](/api_reference/gateway), [configuration](/configuration); tests: `tests/test_features_api.py`, `tests/test_config_api.py`, `tests/test_config_store.py`

Two CI checks keep this page honest. `scripts/check_docs_parity.py` fails when a CLI command or backend route is not mentioned under `docs/`. `scripts/check_feature_chain.py` fails when any registered feature lacks its route, its dashboard call, or its docs mention. The [feature parity matrix](/maintainers/feature-parity) shows the chain per feature.

---

## 📈 Test Coverage

Counts go stale, so this page does not keep them. The [CI workflow](https://github.com/multimindlab/multimind-sdk/blob/develop/.github/workflows/ci.yml) publishes passed, skipped, and failed numbers per Python version on every run. Gates: a 95% pass rate and a 20% line-coverage floor on the full suite, 70% line coverage on the PII guard and the guard proxy, and 35% on the provider, agent, and CLI trees. Tests that need torch, a GPU, a remote vector database, or a provider key skip in the default suite; the integration suite runs with `pytest -m integration`.

---

## 🎯 Summary

### What works today, with tests
- Thirteen first-party providers, vision input, model sessions, and context transfer
- PII guard, cost budgets, hallucination detection, audit trails, and run tracing
- Guard proxy, governance dashboard, AI-usage audit and chargeback, and evidence reports
- Framework adapters for LangChain, LlamaIndex, CrewAI, AutoGen, Haystack, and raw OpenAI clients, tested with stubs
- Anthropic MCP server
- Agents with native function calling, an orchestrator with bounds, and bounded self-evolving exemplars
- GDPR runtime enforcement, HIPAA validation that never fabricates a pass, differential privacy, and regulatory change detection
- RAG with knowledge-graph retrieval, embeddings, and synthetic Q/A generation
- Ten Stable memory types
- Six local vector stores, and a unified wrapper that constructs all 49 real backends
- CLI with eight command groups and one backend API with Swagger

### Real but unproven in CI
- 43 client-backed vector stores that need a live service
- Model conversion, LoRA, QLoRA, adapter tuning, MoE, compression, and the non-transformer models, which need torch
- Six Beta memory types, the reasoning chain, MCP workflows, the pipeline builder, and adaptive privacy

### Placeholders
- Quantum memory and quantum search
- Eleven vector store modules
- Zero-knowledge proofs, homomorphic encryption, federated compliance, self-healing, watermarking, and federated learning
- HyperLoRA, the visual workflow builder, and the advanced monitoring engines

---

> This page replaces marketing claims with what the test suite can show. When a row is wrong, open an issue with the row name; the fix goes into both the code and this page. See the [roadmap](https://github.com/multimindlab/multimind-sdk/blob/develop/ROADMAP.md) for what comes next.
