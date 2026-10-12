---
title: "Governance Dashboard"
---

`multimind dashboard` serves a self-hosted, single-page AI governance dashboard
over the artifacts the SDK already produces: the compliance audit trail
(`AuditLog` JSONL), the cost log (`CostTracker` JSONL), a static shadow-AI
inventory scan of your project, and a guardrails config file that the guard
proxy loads at startup. Everything is read from local files; the dashboard
makes no external calls and nothing leaves the machine.

It requires the gateway extras:

```bash
pip install "multimind-sdk[gateway]"
```

## Layout

```
+-------------+----------------------------------------------------------+
| MultiMind   |  Overview                                                |
| Governance  |                                                          |
|             |  [PII detections] [Blocked] [Total spend] [Models] [Budg]|
| > Overview  |                                                          |
|   Audit     |  +--------------------+  +---------------------------+   |
|   Costs     |  | Spend, last 30 days|  | PII detections by type    |   |
|   Inventory |  |   /\      _/\      |  | email    ############ 42  |   |
|   Guardrails|  |  /  \____/    \_   |  | ssn      ####          9  |   |
|   Scan Text |  +--------------------+  | api_key  ##            4  |   |
|   Evidence  |                          +---------------------------+   |
|   Compliance|                                                          |
| Local       |                                                          |
| artifacts   |                                                          |
| only.       |                                                          |
+-------------+----------------------------------------------------------+
```

Eight views, hash-routed (`#/overview`, `#/audit`, ...):

- **Overview** — stat tiles (PII detections, blocked requests, total spend,
  models in use, budget when configured), a 30-day spend line chart, and a
  PII-by-type bar chart. All-time and last-24h numbers side by side.
- **Audit Trail** — the parsed audit JSONL, newest first, filterable by PII
  type and blocked/allowed status, paginated. Blocked rows are highlighted.
  Records contain types, counts, and hash tags only — never raw PII.
- **Costs** — chargeback table per tag with share bars, plus per-model totals
  (calls, input/output tokens, cost).
- **Inventory** — the `scan_project` shadow-AI report grouped by provider,
  with a red banner when hardcoded API keys are found. Results are cached
  in-process; the Rescan button forces a fresh scan.
- **Guardrails** — a no-code form (redaction strategy, block-on checkboxes
  for the nine PII types, budget ceiling, output scanning) that saves a
  validated `guardrails.json`. After a save the view shows the server's
  `apply_hint`, and an "Apply to the running proxy" card holds a proxy URL
  input (default `http://127.0.0.1:8400`, remembered in the browser's
  `localStorage`) and a **Reload proxy** button that POSTs
  `/v1/guardrails/reload` on that URL. Failures (proxy not running, wrong
  URL, or the browser blocking the cross-origin call) are shown inline with
  the equivalent `curl` command.
- **Scan Text** — paste text, pick a strategy (detect only, `mask`, `hash`,
  `remove`), and `POST /api/scan-text` returns a findings table (type, start,
  end, matched text), per-type counts, and the redacted text when a strategy
  was chosen. Nothing is logged.
- **Evidence Report** — download buttons for `GET /api/evidence-report`
  as HTML or Markdown (opened in a new tab). The Overview also links here.
- **Compliance** — reads `GET /v1/compliance/dashboard` and
  `GET /v1/compliance/alerts` for an organization id: score, passed /
  failed / unverifiable counts, a per-check score trend, recent issues, and
  alerts. It also has a "Record a check" form that POSTs
  `/v1/compliance/monitor` with the organization fields and the text as
  `metadata.text`, shows the raw result, and refreshes the metrics. These
  routes are served by the MultiMind API (`multimind backend`, default
  `http://127.0.0.1:8080`), **not** by the dashboard process. The view has a
  backend URL input (defaults to `MULTIMIND_API_BASE` when set, otherwise
  `http://127.0.0.1:8080`) and an organization id input, both remembered in
  `localStorage`. When the backend is unreachable the view says so and tells
  you to start `multimind backend`. Compliance history lives in the backend's
  memory, so it resets when that process restarts.

Calls from the dashboard page to the proxy or the backend are cross-origin
browser requests. Both servers answer them only when their CORS allow-list
includes the dashboard origin, e.g.
`MULTIMIND_CORS_ORIGINS=http://127.0.0.1:8501 multimind backend` and
`MULTIMIND_CORS_ORIGINS=http://127.0.0.1:8501 multimind serve`
(see [configuration](/configuration)). Without it the browser blocks the
call and the view shows a `curl` fallback. If `MULTIMIND_API_KEY` is
set for the dashboard, the page sends it as `X-API-Key` on these calls too.

The frontend is dependency-free: vanilla ES modules, hand-rolled inline SVG
charts, system font stack, light and dark themes via `prefers-color-scheme`.

## Quickstart

Two terminals:

```bash
# 1. Run the compliance proxy, writing governance artifacts as it works
multimind serve --audit-log audit.jsonl --block-on ssn,credit_card

# 2. Point the dashboard at the same artifacts
multimind dashboard --audit-log audit.jsonl --costs-log costs.jsonl --project .
```

Then open http://127.0.0.1:8501. `python -m multimind.dashboard` works too
(configured via the env vars below).

Note: `multimind serve` writes its cost log to `costs.jsonl` by default
(`--costs-log` / `MULTIMIND_PROXY_COSTS_LOG`). The costs view reads any
`CostTracker(jsonl_path=...)` log, including ones from `track_costs`-wrapped
models in your own code; pass its path with `--costs-log`.

## Configuration

Constructor args / CLI flags override environment variables:

| Setting | Flag | Env var | Default |
| --- | --- | --- | --- |
| Audit log | `--audit-log` | `MULTIMIND_DASHBOARD_AUDIT_LOG` | `audit.jsonl` |
| Costs log | `--costs-log` | `MULTIMIND_DASHBOARD_COSTS_LOG` | `costs.jsonl` |
| Project dir | `--project` | `MULTIMIND_DASHBOARD_PROJECT` | `.` |
| Guardrails file | `--guardrails` | `MULTIMIND_DASHBOARD_GUARDRAILS` | `guardrails.json` |
| Host | `--host` | `MULTIMIND_DASHBOARD_HOST` | `127.0.0.1` |
| Port | `--port` / `-p` | `MULTIMIND_DASHBOARD_PORT` | `8501` |
| Write API key | — | `MULTIMIND_DASHBOARD_API_KEY` | unset (writes unauthenticated) |

Programmatic use:

```python
from multimind.dashboard import DashboardSettings, create_dashboard_app

app = create_dashboard_app(DashboardSettings(audit_log="audit.jsonl"))
# DashboardSettings fields: audit_log, costs_log, project_path, guardrails_path,
# host, port, write_api_key. DashboardSettings.from_env() reads the env vars above.
```

## Endpoint reference

Interactive docs at `/docs` (Swagger UI). All endpoints return JSON except
`GET /api/evidence-report` (a Markdown/HTML download); only
`PUT /api/guardrails` writes anything.

| Endpoint | Description |
| --- | --- |
| `GET /api/summary` | Headline numbers: PII events by type, blocked count, spend, models, budget status; all-time and last-24h. |
| `GET /api/audit?limit&offset&type&blocked` | Parsed audit records, newest first. `type` filters by PII type, `blocked` by status. |
| `GET /api/costs?period` | Chargeback by tag (`share_pct` included), per-model totals, and a daily spend series. `period=2026-07` filters by ISO timestamp prefix, as `multimind audit costs --period` does. |
| `GET /api/inventory?refresh&include_env` | Shadow-AI scan of the project dir; cached until `refresh=true`. `include_env=true` also reports which known AI env keys are set (names only, never values), as `multimind audit scan --include-env`, and is never cached. |
| `POST /api/scan-text` | Detect PII in posted `text` (findings with type, span, and matched text; counts); optional `strategy` (`mask`/`hash`/`remove`) also returns `redacted`. Results go only to the caller; nothing is logged. |
| `GET /api/evidence-report?format&period&organization&project&costs_log` | Compliance evidence report as an attachment; `format=html` (default) or `md`. Built from the audit log, costs log, and cached inventory scan; absent artifacts are stated as absent. `period` and `organization` mirror the `multimind compliance report-evidence` flags; `project` scans that directory instead of the default (400 if it is not a directory); `costs_log` uses that JSONL file instead of the configured one (400 if missing). |
| `GET /api/guardrails` | Current guardrails config (defaults if the file does not exist yet). |
| `PUT /api/guardrails` | Validate and write the guardrails file; 422 on invalid strategy, unknown PII type, non-positive budget, or unknown fields; 401 when `MULTIMIND_DASHBOARD_API_KEY` is set and the `X-API-Key` header does not match. |
| `GET /health`, `GET /ready` | Liveness and readiness probes. |
| `GET /` | The dashboard UI (static files). |

## Guardrails workflow

The Guardrails view writes a JSON file with this schema:

```json
{
  "strategy": "mask",
  "block_on": ["ssn", "credit_card"],
  "budget_max_cost": 25.0,
  "scan_output": true
}
```

`multimind serve` writes the cost log (`costs.jsonl` by default) so the
Costs view has rows after proxied calls.

To apply a guardrails change to a running `multimind serve` process:

1. Edit and save in the dashboard (or `PUT /api/guardrails`).
2. `POST /v1/guardrails/reload` on the proxy. The reload endpoint re-reads
   the path in the proxy's `MULTIMIND_DASHBOARD_GUARDRAILS` env var (default
   `guardrails.json` in the proxy's working directory), not the `--config`
   path, so keep the two pointing at the same file. Otherwise restart with
   `multimind serve --config guardrails.json`, which is what the dashboard's
   save message suggests.

`--config` loads the file via `ProxySettings.from_file`: `budget_max_cost`
maps to the proxy `budget`, file values override `MULTIMIND_PROXY_*` env
vars, and explicit CLI flags override the file. Reload keeps the process
host, port, cost log, and audit log.

Budget status on the Overview compares the configured ceiling against all
logged spend in the costs log. The proxy's own `Budget` enforcement is per
session, so treat the dashboard number as reporting, not enforcement.

## Security

- Binds `127.0.0.1` by default; only local processes can reach it.
- Read-only over local artifacts: the sole write path is the guardrails file,
  and the server never calls out to any network service.
- The underlying logs contain PII types, counts, and hash tags — never raw
  PII, prompts, or responses.
- Read endpoints have no authentication. The one write endpoint
  (`PUT /api/guardrails`) can be gated by setting `MULTIMIND_DASHBOARD_API_KEY`;
  callers must then send a matching `X-API-Key` header. For team access, keep
  the default localhost bind and put a reverse proxy with TLS and auth (e.g.
  nginx with basic auth or your SSO proxy) in front; do not expose the port
  directly.
