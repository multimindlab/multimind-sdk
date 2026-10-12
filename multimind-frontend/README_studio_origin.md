# MultiMind Studio

Private SaaS frontend and platform server for the open-source
[MultiMind SDK](https://github.com/multimindlab/multimind-sdk): governance
dashboard (audit trail, costs, shadow-AI inventory, guardrails, PII scan,
evidence reports), a chat UI backed by the gateway, and a LangSmith-style
observability platform (projects, traces with span waterfalls, feedback,
monitoring charts) fed by the SDK's `multimind.observability.tracing` client.

## Prerequisites

- Node.js 18.x or newer (built and tested with Node 18.20.4, npm 10.7.0)
- The MultiMind Python SDK: `pip install multimind-sdk[gateway]`
  (or an editable install of the SDK repo, branch `bug-fix-overall-feature-develop`)

## Run it

Start the backends in separate terminals:

```sh
multimind dashboard                                   # governance API on http://localhost:8501
multimind serve                                       # gateway API on http://localhost:8000
MULTIMIND_PLATFORM_KEY=dev-key python -m backend.app  # platform API on http://localhost:8600 (from this repo root)
```

Then start the frontend (from this repo root):

```sh
npm install
npm run dev           # http://localhost:5173
```

Each page degrades gracefully when its backend is down and tells you which
command starts it, so you can also run with only one backend up.

## How the proxy works

The Vite dev server proxies API calls so the browser only ever talks to one
origin and no CORS configuration is needed on either backend:

| Frontend path      | Proxied to                                            |
| ------------------ | ----------------------------------------------------- |
| `/api/*`           | `http://localhost:8501` (dashboard)                   |
| `/db-health`       | `http://localhost:8501/health`                        |
| `/v1/*`            | `http://localhost:8000` (gateway)                     |
| `/gw-health`       | `http://localhost:8000/health`                        |
| `/platform/*`      | `http://localhost:8600` (platform, prefix stripped)   |
| `/platform-health` | `http://localhost:8600/health`                        |

See `vite.config.ts`. For a production deployment, replicate these routes in
your reverse proxy (nginx, Caddy, etc.) in front of the built `dist/`.

## Pages

- **Overview** - stat tiles (PII events, blocked, spend, models in use), 30-day
  spend line chart, PII-by-type bar chart
- **Audit Trail** - PII events with type and blocked filters, pagination,
  blocked rows highlighted
- **Costs** - chargeback-by-tag table with share bars, per-model table, daily
  spend chart
- **Inventory** - shadow-AI findings grouped by provider, risk banner when
  hardcoded keys are found, rescan button
- **Guardrails** - edit the guardrails config (redaction strategy, block-on PII
  types, budget ceiling, output scanning); saving shows the
  `multimind serve --config ...` restart hint
- **Scan** - paste text, pick a redaction strategy, see findings and the
  redacted result (`POST /api/scan-text`)
- **Evidence** - download the evidence report as Markdown or HTML, or preview
  the HTML inline in a sandboxed iframe
- **Chat** - talk to a model through the gateway (`POST /v1/chat`), with the
  model list from `GET /v1/models`

The sidebar shows live health for all three backends (polled every 15 seconds).

## Observability platform

The **Platform** sidebar section is a LangSmith-style UI over the trace
platform server in `backend/` (port 8600). Start it with an API key:

```sh
cd multimind-frontend
MULTIMIND_PLATFORM_KEY=dev-key python -m backend.app   # http://localhost:8600
```

- **Projects** - every tracing project with run count, last activity, total
  cost, and an error-rate bar; click a project to open its runs
- **Traces** - runs table (status, name, run-type chip, model, latency,
  tokens, cost, feedback, time) with project/status/run-type filters, search,
  pagination, and a 10-second auto-refresh toggle
- **Trace detail** (`/traces/:id`) - run header (status, latency, cost,
  tokens, model, tags), a span waterfall scaled by start/end times and
  indented by parent depth, a click-to-open span side panel (chars, hashes,
  inputs/outputs when recorded), and a thumbs up/down + comment feedback
  widget
- **Monitoring** - per-project 30-day charts from `/api/v1/stats`: runs/day
  stacked with errors, cost/day, p50/p95 latency, and token volume

### Login flow

Every platform API call needs the `X-API-Key` header. The first visit to a
platform page shows a sign-in form; the key you enter is verified against
`GET /api/v1/projects` and, if accepted, kept in `sessionStorage`
(`mm_platform_key`) for the rest of the browser session. A "Sign out" button
appears in the sidebar while signed in, and any 401 from the server clears the
stored key and returns you to the sign-in form. Governance and Gateway pages
are not gated.

**Honest single-tenant note:** the platform server has exactly one shared API
key (`MULTIMIND_PLATFORM_KEY`) for the whole deployment. There are no user
accounts, roles, or per-project permissions - anyone with the key sees and can
annotate every trace. Treat it as a self-hosted, single-team tool.

## Tests and build

```sh
npm test -- --run     # vitest + Testing Library; all network calls are mocked
npm run build         # tsc type-check + vite production build into dist/
npm run preview       # serve the production build locally
```

## Honest note about Chat

The Chat page can only return replies if the gateway process has at least one
provider configured (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or an
`OLLAMA_API_BASE`) in its environment. Without a key the gateway answers 503
("No models are properly configured") and the page shows that error with setup
instructions instead of a conversation. The frontend itself never handles or
stores provider keys.
