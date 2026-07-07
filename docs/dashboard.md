# Governance Dashboard

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
|             |  +--------------------+  | api_key  ##            4  |   |
| Local       |                          +---------------------------+   |
| artifacts   |                                                          |
| only.       |                                                          |
+-------------+----------------------------------------------------------+
```

Five views, hash-routed (`#/overview`, `#/audit`, ...):

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
  validated `guardrails.json`.

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

Note: `multimind serve` keeps its cost tracker in memory. The costs view reads
a `CostTracker(jsonl_path=...)` log, e.g. from `track_costs`-wrapped models in
your own code; pass its path with `--costs-log`.

## Configuration

Constructor args / CLI flags override environment variables:

| Setting | Flag | Env var | Default |
| --- | --- | --- | --- |
| Audit log | `--audit-log` | `MULTIMIND_DASHBOARD_AUDIT_LOG` | `audit.jsonl` |
| Costs log | `--costs-log` | `MULTIMIND_DASHBOARD_COSTS_LOG` | `costs.jsonl` |
| Project dir | `--project` | `MULTIMIND_DASHBOARD_PROJECT` | `.` |
| Guardrails file | `--guardrails` | `MULTIMIND_DASHBOARD_GUARDRAILS` | `guardrails.json` |
| Host | `--host` | `MULTIMIND_DASHBOARD_HOST` | `127.0.0.1` |
| Port | `--port` | `MULTIMIND_DASHBOARD_PORT` | `8501` |

Programmatic use:

```python
from multimind.dashboard import DashboardSettings, create_dashboard_app

app = create_dashboard_app(DashboardSettings(audit_log="audit.jsonl"))
```

## Endpoint reference

Interactive docs at `/docs` (Swagger UI). All endpoints return JSON except
`GET /api/evidence-report` (a Markdown/HTML download); only
`PUT /api/guardrails` writes anything.

| Endpoint | Description |
| --- | --- |
| `GET /api/summary` | Headline numbers: PII events by type, blocked count, spend, models, budget status; all-time and last-24h. |
| `GET /api/audit?limit&offset&type&blocked` | Parsed audit records, newest first. `type` filters by PII type, `blocked` by status. |
| `GET /api/costs` | Chargeback by tag (`share_pct` included), per-model totals, and a daily spend series. |
| `GET /api/inventory?refresh` | Shadow-AI scan of the project dir; cached until `refresh=true`. |
| `POST /api/scan-text` | Detect PII in posted `text` (findings, counts); optional `strategy` (`mask`/`hash`/`remove`) also returns `redacted`. Results go only to the caller; nothing is logged. |
| `GET /api/evidence-report?format` | Compliance evidence report as an attachment; `format=html` (default) or `md`. Built from the audit log, costs log, and cached inventory scan; absent artifacts are stated as absent. |
| `GET /api/guardrails` | Current guardrails config (defaults if the file does not exist yet). |
| `PUT /api/guardrails` | Validate and write the guardrails file; 422 on invalid strategy, unknown PII type, non-positive budget, or unknown fields. |
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

The guard proxy reads its policy at startup, so applying a change is a
restart:

1. Edit and save in the dashboard (or `PUT /api/guardrails`).
2. Restart the proxy with `multimind serve --config guardrails.json`.

`--config` loads the file via `ProxySettings.from_file`: `budget_max_cost`
maps to the proxy `budget`, file values override `MULTIMIND_PROXY_*` env
vars, and explicit CLI flags override the file. A running proxy does not
watch the file; changes take effect only on restart.

Budget status on the Overview compares the configured ceiling against all
logged spend in the costs log. The proxy's own `Budget` enforcement is per
session, so treat the dashboard number as reporting, not enforcement.

## Security

- Binds `127.0.0.1` by default; only local processes can reach it.
- Read-only over local artifacts: the sole write path is the guardrails file,
  and the server never calls out to any network service.
- The underlying logs contain PII types, counts, and hash tags — never raw
  PII, prompts, or responses.
- There is no authentication. For team access, keep the default localhost
  bind and put a reverse proxy with TLS and auth (e.g. nginx with basic auth
  or your SSO proxy) in front; do not expose the port directly.
