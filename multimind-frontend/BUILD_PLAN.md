# MultiMind Frontend — Build Plan

Private React frontend for the MultiMind SDK. This folder is gitignored in the SDK repo
(`.gitignore` → `multimind-frontend/`) and must never be pushed here.

## 0. Existing asset — do not start from zero

A complete v1 of this frontend already exists in the separate private repo
`/Users/darshankumar/Daemongodwiz/Ai2innovate/multimindsdk-studio`
(package name is literally `multimind-frontend`): Vite 5 + React 18 + TypeScript,
react-router-dom, hand-rolled SVG charts, vitest suites, and these pages already wired:
Overview, Chat, Guardrails, Scan, Costs, Audit, Inventory, Evidence, plus
LangSmith-style platform pages (projects / traces / span waterfall / monitoring) with
API-key login. Its Vite proxy already targets the dashboard, gateway, and platform server.

**Recommended bootstrap:** copy `multimindsdk-studio` frontend sources into this folder
(or keep developing in the studio repo and treat this folder as a working checkout),
then extend per the roadmap below. Rebuilding from scratch duplicates ~9 finished pages
and 65+ passing tests.

## 1. Stack

- Vite 5 + React 18 + TypeScript, `react-router-dom` for routing
- No heavy UI/chart deps (hand-rolled SVG charts already exist); add a component lib only if design demands it
- vitest + @testing-library/react
- Dev-server proxy for all backend calls → no CORS changes needed in the SDK

## 2. Backend targets (from API surface audit, 2026-07-14)

| App | Launch | Port (default) | Auth |
|---|---|---|---|
| Governance dashboard | `multimind dashboard` | 8501 (127.0.0.1) | none (localhost-only) |
| Guard proxy (OpenAI-compatible) | `multimind serve` | 8400 | none; forwards `Authorization` upstream |
| API gateway (`gateway/api.py`) | `uvicorn multimind.gateway.api:app` | 8000 | none |
| Compliance router | mounted in gateway at `/v1/compliance/*` | 8000 | none |
| RAG API (`gateway/rag_api.py`) | `uvicorn multimind.gateway.rag_api:app` | 8000 (conflicts with gateway — run on 8100) | JWT `/token` or `X-API-Key` |
| Unified / multi-model APIs (`multimind/api/`) | uvicorn manual | 8000 default | optional `X-API-Key` |
| MCP server | stdio only | — | not browser-consumable |

Vite proxy plan (extends the studio one):

```
/api        → 127.0.0.1:8501   # dashboard JSON API
/v1/chat*   → 127.0.0.1:8000   # gateway chat/sessions/models/metrics/compare
/v1/compliance → 127.0.0.1:8000
/proxy/v1   → 127.0.0.1:8400   # guard proxy (OpenAI-compatible playground)
/rag        → 127.0.0.1:8100   # rag_api, prefix stripped
```

CORS stays off in the SDK; the dev proxy (and later a production reverse proxy /
static-mount) handles origins. Dashboard has no CORS middleware at all — do not call it
cross-origin directly.

## 3. Feature → page → endpoint map

### Phase A — already covered by existing endpoints (port of studio v1)
| Page | Endpoints |
|---|---|
| Overview (health, 24h audit stats, spend, budget) | `GET /api/summary`, `/health` of each backend |
| PII Audit log (filter/paginate) | `GET /api/audit?limit&offset&type&blocked` |
| Costs & chargeback (by tag/model/day) | `GET /api/costs` |
| Shadow-AI Inventory | `GET /api/inventory?refresh` |
| Ad-hoc PII Scan | `POST /api/scan-text` |
| Evidence report (download HTML/MD) | `GET /api/evidence-report?format=` |
| Guardrails no-code editor | `GET/PUT /api/guardrails` |
| Chat playground (streaming, PII-guarded) | guard proxy `POST /v1/chat/completions` (SSE), `GET /v1/models` |

### Phase B — endpoints exist but unused by v1 (new pages)
| Page | Endpoints |
|---|---|
| Model catalog + health + metrics | gateway `GET /v1/models`, `/v1/metrics`, `POST /v1/health/check` |
| Model compare (side-by-side generations) | gateway `POST /v1/compare`, `/v1/generate` |
| Chat sessions manager | gateway `POST/GET/DELETE /v1/sessions*` |
| Compliance center (monitor, reports, regulations, alerts) | `/v1/compliance/monitor`, `/report`, `/regulations`, `/alerts*`, `/dashboard` |
| RAG workbench (upload docs, query, generate, switch model) | rag_api `POST /token`, `/documents`, `/files`, `/query`, `/generate`, `GET /documents/count` |

### Phase C — needs new backend endpoints first (SDK is Python/CLI-only here)
Build thin FastAPI routers in the SDK (or extend the dashboard app) before UI work:
1. **Tracing/observability** — expose RunTracer SQLite sink (or reuse the studio
   platform server on :8600, which already does projects/runs/feedback/stats).
2. **Evaluation** — REST wrapper for `detect_hallucinations` / metric runs.
3. **Model lifecycle** — download/export/delete (today CLI `models` only).
4. **Model conversion** (`convert`, torch-gated) and **context transfer** — job-style
   endpoints (POST job → poll status) since they're long-running.
5. **Fine-tuning** (LoRA/QLoRA) — job-style endpoints; torch-gated, keep behind a
   feature flag in the UI.
6. **Cost recording** — write-side endpoint if non-proxy apps should report spend.

Do NOT build UI for honestly-Planned SDK features (ZKP, homomorphic encryption,
watermarking, networked RAFT, quantum) — show them as "Planned" badges at most.

## 4. App structure

```
multimind-frontend/
  src/
    api/            # one client per backend: dashboard.ts, gateway.ts, guardProxy.ts, rag.ts
    components/     # charts (SVG), tables, status badges, BackendDown banners
    pages/          # one folder per Phase A/B page
    auth/           # rag_api JWT login + token storage; X-API-Key support
    test/
  vite.config.ts    # proxy table above
  package.json
```

Conventions carried from v1: every page handles backend-down gracefully
(BackendStatus/BackendDown components), empty-state components, no external chart libs,
all API clients unit-tested against fixture JSON.

## 5. Auth strategy

- Dev/local: none needed (all backends localhost, proxied).
- RAG pages: JWT via `POST /token`, kept in memory (not localStorage), auto-refresh on 401.
- Production/team deploy: single reverse proxy (nginx/caddy) in front of all apps with
  one auth layer (basic auth or OIDC) — the SDK apps' auth is inconsistent by design
  (dashboard/gateway/proxy have none), so auth belongs at the edge.

## 6. Milestones

1. **M1 — Bootstrap (0.5 day):** copy studio v1 here, `npm install`, verify all Phase A
   pages against live `multimind dashboard` + `multimind serve`; `node_modules/`/`dist/`
   ignored inside this folder too.
2. **M2 — Gateway pages (1–2 days):** model catalog/metrics/compare, sessions manager.
3. **M3 — Compliance center (1 day):** compliance router pages + alerts config.
4. **M4 — RAG workbench (1–2 days):** JWT flow, upload, query, generate, model switch;
   requires deciding rag_api's launch port (add `multimind rag-serve --port 8100`?).
5. **M5 — Phase C backend endpoints + pages:** tracing first (or reuse studio platform
   server), then evaluation, then job-style convert/fine-tune. Scope per demand.

## 7. Privacy rules for this folder

- `multimind-frontend/` is in the SDK repo's `.gitignore`; never `git add -f` it.
- If it should have its own history, `git init` inside it and push to a **private** remote
  (or keep it inside multimindsdk-studio, which is already a separate repo).
- No SDK-repo code may import from here; the coupling is HTTP-only.
