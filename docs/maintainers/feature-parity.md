---
title: "Feature Parity Matrix"
description: "Which surface (SDK, CLI, backend API, dashboard, docs) exposes each MultiMind feature, and the remaining gaps."
---

Audited and standardised 2026-10-12 against version 1.0.0. The rule this page tracks: a feature that works in the CLI must also be reachable from `multimind backend`, visible in the governance dashboard where a UI makes sense, and documented here. A dash means the surface does not exist. CI runs `scripts/check_docs_parity.py`, which fails when a CLI command or a backend route is not named anywhere under `docs/`.

The API column lists routes on `multimind backend` (see [openapi-platform.json](https://github.com/multimindlab/multimind-sdk/blob/develop/docs/api/openapi-platform.json)). The dashboard has eight views: overview, audit, costs, inventory, guardrails, scan text, evidence report, compliance.

## Matrix

| Feature | SDK module | CLI | Backend API | Dashboard | Docs | Status |
|---|---|---|---|---|---|---|
| PII scan / redact | `compliance/guard.py` | `compliance scan-text` | `POST /api/scan-text` | Scan text view | [CLI](/cli), [dashboard](/dashboard) | Stable |
| Guard wrapper `guard(model)` | `compliance/guard.py` | — (SDK) | — (via proxy) | — | [compliance](/compliance), [integrations](/integrations) | Stable |
| OpenAI-compatible guard proxy | `gateway/guard_proxy.py` | `serve` | `/v1/chat/completions`, `/v1/embeddings`, `/v1/guardrails/reload` | Reload button in Guardrails view | [guard proxy](/guard-proxy) | Stable |
| Guardrails file | `dashboard/server.py` | `serve --config` | `GET/PUT /api/guardrails` | Guardrails view | [dashboard](/dashboard) | Stable |
| PII audit trail | `compliance/guard.py` `AuditLog` | — | `GET /api/audit`, `GET /api/summary` | Audit, Overview | [dashboard](/dashboard) | Stable |
| Cost tracking, budgets, chargeback | `observability/cost_tracker.py` | `audit costs --period` | `GET /api/costs?period` | Costs, Overview | [AI inventory](/ai-inventory) | Stable |
| Shadow-AI inventory | `observability/ai_inventory.py` | `audit scan --include-env` | `GET /api/inventory?include_env` | Inventory | [AI inventory](/ai-inventory) | Stable |
| Evidence report | `compliance/reporting.py` | `compliance report-evidence` | `GET /api/evidence-report?format&period&organization&project&costs_log` | Evidence report view | [compliance](/compliance), [dashboard](/dashboard) | Stable |
| Compliance monitor and report | `gateway/compliance_api.py` | `compliance run-compliance`, `generate-report` | `POST /v1/compliance/monitor`, `/report`, `GET /regulations`, `/healthcare/use-cases` | Compliance view (record a check) | [compliance](/compliance) | Stable |
| Compliance dashboard and alerts | `gateway/compliance_api.py` (per-process store) | `compliance dashboard`, `alerts`, `configure-alerts` | `GET /v1/compliance/dashboard`, `/alerts`, `POST /alerts/configure` | Compliance view | [compliance](/compliance) | Beta |
| GDPR runtime policy | `compliance/gdpr.py` | via `run-compliance` with `metadata.purpose` | via `/v1/compliance/monitor` | Compliance view | [compliance](/compliance) | Stable |
| Regulatory watcher | `compliance/regulatory_watch.py` | — (SDK) | — | — | [compliance](/compliance) | Stable |
| Differential privacy | `compliance/advanced.py` `DPMechanism` | — (SDK) | — | — | [compliance quickstart](/compliance_quickstart) | Stable |
| Framework adapters | `integrations/frameworks` | — (SDK) | — | — | [integrations](/integrations) | Stable |
| Hallucination detection | `evaluation/hallucination.py` | — (SDK) | — (MCP `check_grounding`) | — | [cookbook](/cookbook), [feature status](/feature-status) | Stable |
| Run tracing | `observability/tracing.py` | — (SDK) | — | — | [tracing](/tracing) | Stable |
| `ModelSession` switching | `client/model_session.py` | — (SDK) | — | — | [cookbook](/cookbook) | Stable |
| Agents and orchestrator | `agents/` | — (SDK) | — | — | [architecture](/architecture) | Stable / Beta |
| Chat | `gateway/models.py` | `chat start` | `POST /v1/chat`, `/v1/generate` | — | [CLI](/cli), [gateway](/api_reference/gateway) | Stable |
| Chat sessions | `gateway/chat.py` | `chat list-sessions`, `load`, `save`, `delete` | `/v1/sessions`, `/{id}`, `/{id}/messages`, `/{id}/save`, `/load` | — | [gateway](/api_reference/gateway) | Stable |
| Model compare, health, metrics, list | `gateway/` | `models compare`, `health`, `metrics`, `list` | `POST /v1/compare`, `POST /v1/health/check`, `GET /v1/metrics`, `GET /v1/models` | — | [CLI](/cli), [gateway](/api_reference/gateway) | Stable |
| Local model download, export, delete | transformers directly | `models download`, `export`, `delete` | — (CLI only, needs torch) | — | [CLI](/cli) | Beta |
| Model conversion | `model_conversion` | `convert` (needs `[finetune]`) | — (CLI only, needs torch) | — | [CLI](/cli) | Beta |
| Context transfer | `context_transfer` | `context-transfer` | — (CLI only) | — | [CLI](/cli) | Stable |
| RAG | `rag/`, `gateway/rag_api.py` | — (SDK) | `/documents`, `/files`, `/query`, `/generate`, `/models/switch`, `/token` | — | [RAG](/rag), [RAG API](/api_reference/rag_api) | Stable |
| Memory systems | `memory/` | — (SDK) | — | — | [memory](/memory) | Stable |
| Fine-tuning | `fine_tuning/` | — (SDK) | — | — | [features](/features_and_functions) | Beta |
| MCP server | `mcp_server/` | `python -m multimind.mcp_server` | — | — | [MCP server](/mcp-server) | Stable |
| Backend server | `server/platform.py` | `backend` | `/`, `/health`, `/ready`, `/docs` | — | [API overview](/api_reference/overview) | Stable |

## Remaining gaps, by decision

- **CLI only, by design:** `models download`, `models export`, `models delete`, `convert`, and `context-transfer` need torch or local files and are not server features. The CLI page says so.
- **SDK only, by design:** the guard wrapper, regulatory watcher, differential privacy, framework adapters, hallucination detection, tracing, `ModelSession`, agents, memory, and fine-tuning are Python APIs. They have docs pages and tests but no CLI or route.
- **No dashboard view** for chat, sessions, models, and RAG routes. The governance dashboard covers governance artifacts only. A chat playground lives in the separate frontend workspace.
- **Compliance history is per process.** `/v1/compliance/dashboard` and `/alerts` reflect the `POST /v1/compliance/monitor` calls recorded in that backend process and reset on restart. From the CLI each command is its own process, so `compliance dashboard` prints nothing recorded. The durable artifacts remain the JSONL audit and cost logs.
- **Cross-origin calls** from the dashboard page to the backend and the proxy need `MULTIMIND_CORS_ORIGINS` set on both servers. The views show a `curl` fallback when blocked.
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
