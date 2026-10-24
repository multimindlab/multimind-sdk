---
title: "Feature Parity Matrix"
description: "Which surface (SDK, CLI, backend API, dashboard, docs) exposes each MultiMind feature, and the remaining gaps."
---

Audited 2026-10-12 and standardised 2026-10-13 against version 1.0.0. The rule this page tracks: every component that works must be reachable through the chain component, backend route, dashboard view, docs page, with a test for each link. A dash means the surface does not exist by decision. Two CI checks enforce it: `scripts/check_docs_parity.py` fails when a CLI command or a backend route is not named under `docs/`, and `scripts/check_feature_chain.py` fails when any feature in the `GET /v1/features` registry has a route missing from the backend, a route the dashboard never calls, or a route no docs page mentions. Both run on every push.

The API column lists routes on `multimind backend` (see [openapi-platform.json](https://github.com/multimindlab/multimind-sdk/blob/develop/docs/api/openapi-platform.json)). The dashboard has 26 views in three groups: Playground (features, models, chat, compare, RAG, hallucination, agents, memory, vector stores, fine-tuning, conversion, regulatory watch, differential privacy, MCP tools, integrations, tracing, context transfer), Governance (overview, audit, costs, inventory, guardrails, scan text, evidence report, compliance), and Settings.

## Matrix

| Feature | SDK module | CLI | Backend API | Dashboard | Docs | Status |
|---|---|---|---|---|---|---|
| PII scan / redact | `compliance/guard.py` | `compliance scan-text` | `POST /api/scan-text` | Scan text view | [CLI](/cli), [dashboard](/dashboard) | Stable |
| Guard wrapper `guard(model)` | `compliance/guard.py` | — (SDK) | — (via proxy) | — | [compliance](/compliance), [integrations](/integrations) | Stable |
| OpenAI-compatible guard proxy | `gateway/guard_proxy.py` | `serve` | `/v1/chat/completions`, `/v1/embeddings`, `/v1/guardrails/reload` | Chat view (guarded toggle, embed); reload button in Guardrails view | [guard proxy](/guard-proxy) | Stable |
| Guardrails file | `dashboard/server.py` | `serve --config` | `GET/PUT /api/guardrails` | Guardrails view | [dashboard](/dashboard) | Stable |
| PII audit trail | `compliance/guard.py` `AuditLog` | — | `GET /api/audit`, `GET /api/summary` | Audit, Overview | [dashboard](/dashboard) | Stable |
| Cost tracking, budgets, chargeback | `observability/cost_tracker.py` | `audit costs --period` | `GET /api/costs?period` | Costs, Overview | [AI inventory](/ai-inventory) | Stable |
| Shadow-AI inventory | `observability/ai_inventory.py` | `audit scan --include-env` | `GET /api/inventory?include_env` | Inventory | [AI inventory](/ai-inventory) | Stable |
| Evidence report | `compliance/reporting.py` | `compliance report-evidence` | `GET /api/evidence-report?format&period&organization&project&costs_log` | Evidence report view | [compliance](/compliance), [dashboard](/dashboard) | Stable |
| Compliance monitor and report | `gateway/compliance_api.py` | `compliance run-compliance`, `generate-report` | `POST /v1/compliance/monitor`, `/report`, `GET /regulations`, `/healthcare/use-cases` | Compliance view (record a check, generate report, regulations, use cases) | [compliance](/compliance) | Stable |
| Compliance dashboard and alerts | `gateway/compliance_api.py` (per-process store) | `compliance dashboard`, `alerts`, `configure-alerts` | `GET /v1/compliance/dashboard`, `/alerts`, `POST /alerts/configure` | Compliance view (metrics, alerts, alert rules form) | [compliance](/compliance) | Beta |
| GDPR runtime policy | `compliance/gdpr.py` | via `run-compliance` with `metadata.purpose` | via `/v1/compliance/monitor` | Compliance view | [compliance](/compliance) | Stable |
| Regulatory watcher | `compliance/regulatory_watch.py` | — (SDK) | — | Regulatory watch view | [compliance](/compliance) | Stable |
| Differential privacy | `compliance/advanced.py` `DPMechanism` | — (SDK) | — | Differential privacy view | [compliance quickstart](/compliance_quickstart) | Stable |
| Framework adapters | `integrations/frameworks` | — (SDK) | — | Integrations view | [integrations](/integrations) | Stable |
| Hallucination detection | `evaluation/hallucination.py` | — (SDK) | — (MCP `check_grounding`) | Hallucination view | [cookbook](/cookbook), [feature status](/feature-status) | Stable |
| Run tracing | `observability/tracing.py` | — (SDK) | — | Tracing view | [tracing](/tracing) | Stable |
| `ModelSession` switching | `client/model_session.py` | — (SDK) | — | Chat view (session switch) | [cookbook](/cookbook) | Stable |
| Agents and orchestrator | `agents/` | — (SDK) | — | Agents view | [architecture](/architecture) | Stable / Beta |
| Chat | `gateway/models.py` | `chat start` | `POST /v1/chat`, `/v1/generate` | Chat view (chat and generate modes) | [CLI](/cli), [gateway](/api_reference/gateway) | Stable |
| Chat sessions | `gateway/chat.py` | `chat list-sessions`, `load`, `save`, `delete` | `/v1/sessions`, `/{id}`, `/{id}/messages`, `/{id}/save`, `/load` | Chat view (sessions panel) | [gateway](/api_reference/gateway) | Stable |
| Model compare, health, metrics, list | `gateway/` | `models compare`, `health`, `metrics`, `list` | `POST /v1/compare`, `POST /v1/health/check`, `GET /v1/metrics`, `GET /v1/models` | Compare view; Models view (list, metrics, health check) | [CLI](/cli), [gateway](/api_reference/gateway) | Stable |
| Local model download, export, delete | transformers directly | `models download`, `export`, `delete` | — (CLI only, needs torch) | — | [CLI](/cli) | Beta |
| Model conversion | `model_conversion` | `convert` (needs `[finetune]`) | — (CLI only, needs torch) | Conversion view (capability report) | [CLI](/cli) | Beta |
| Context transfer | `context_transfer` | `context-transfer` | — (CLI only) | Context transfer view | [CLI](/cli) | Stable |
| RAG | `rag/`, `gateway/rag_api.py` | — (SDK) | `/documents`, `/files`, `/query`, `/generate`, `/models/switch`, `/token` | RAG view | [RAG](/rag), [RAG API](/api_reference/rag_api) | Stable |
| Memory systems | `memory/` | — (SDK) | — | Memory view | [memory](/memory) | Stable |
| Fine-tuning | `fine_tuning/` | — (SDK) | — | Fine-tuning view (capabilities, synthetic Q/A) | [features](/features_and_functions) | Beta |
| MCP server | `mcp_server/` | `python -m multimind.mcp_server` | — | MCP tools view | [MCP server](/mcp-server) | Stable |
| Backend server | `server/platform.py` | `backend` | `/`, `/health`, `/ready`, `/docs` | Features view (landing), Settings view | [API overview](/api_reference/overview) | Stable |

## Remaining gaps, by decision

- **CLI only, by design:** `models download`, `models export`, `models delete`, `convert`, and `context-transfer` need torch or local files and are not server features. The CLI page says so.
- **No CLI command, by design:** the guard wrapper, framework adapters, `ModelSession`, agents, memory, hallucination detection, tracing, differential privacy, and the regulatory watcher are Python APIs. Each now has a backend route and a dashboard view; a CLI wrapper would add nothing a developer cannot do from Python.
- **Model-backed views need a provider key.** Chat, compare, agents, synthetic Q/A, session switch with the `summary` strategy, and the hallucination judge call a real model, so the views show the backend's "model not available" error until a key is set in the environment. Keys are never entered in the UI.
- **Torch-gated features report, not run.** The fine-tuning and conversion views show the capability report from the backend, including the install hint, because the default install has no torch.
- **Compliance history is per process.** `/v1/compliance/dashboard` and `/alerts` reflect the `POST /v1/compliance/monitor` calls recorded in that backend process and reset on restart. From the CLI each command is its own process, so `compliance dashboard` prints nothing recorded. The durable artifacts remain the JSONL audit and cost logs.
- **Standalone dashboard still needs CORS.** Under `multimind backend` the UI is served at `/` and every call is same-origin. Under the standalone `multimind dashboard`, calls to the backend and the proxy need `MULTIMIND_CORS_ORIGINS` on those servers, and the views show a `curl` fallback when blocked.
- **`models metrics`** prints empty tables from the CLI because the gateway monitor is per process. Documented on the CLI page.

## Fixed on 2026-10-12

- `multimind compliance run-compliance` no longer exits 1 after printing results.
- `multimind --version` works.
- Compliance dashboard and alerts record checks instead of rebuilding an empty shard per request.
- `compliance run-example` names the real missing import.
- Backend gained `period`, `include_env`, and the evidence-report filters, plus session save and load routes.
- Dashboard gained scan text, evidence report, compliance, and proxy reload views.
- Guard proxy accepts `MULTIMIND_CORS_ORIGINS`.
- Docs gained the tracing page, full `convert` and `context-transfer` references, and eleven feature-status rows.

## Standardised on 2026-10-13

- Every SDK component has a backend route: 13 new route groups under `/v1` (evaluation, session switch, context transfer, agents, memory, vector stores, fine-tuning, conversion, regulatory check, differential privacy, MCP tools, integrations, tracing) and a `GET /v1/features` registry that reports each component's availability.
- Every setting except API keys is managed from the dashboard through `GET/PUT /api/config` and `POST /api/config/test-provider`, backed by a JSON config store that wins over environment variables without a restart.
- `multimind backend` serves the dashboard at `/`, so one process is the whole playground with no cross-origin calls. The `/config.js` it serves never embeds a key; the page keeps the API key in the browser.
- The dashboard has 26 views, one per feature, each calling the real route and showing 401, 404, 501, and network errors visibly.
- `scripts/check_feature_chain.py` proves the four links for all 27 registered features in CI.
