# Guard Proxy: `multimind serve`

A one-command, OpenAI-compatible compliance proxy. Point any existing app's
OpenAI `base_url` at it and instantly get PII redaction, budget enforcement,
and an audit trail — with **zero code changes** to your app.

```
Your app (OpenAI SDK) ──> multimind serve ──> upstream (OpenAI, Groq, Ollama, ...)
                          │
                          ├─ redacts PII in requests (mask/hash/remove)
                          ├─ blocks requests carrying forbidden PII types (400)
                          ├─ enforces a spend ceiling (429 when exceeded)
                          ├─ scans model output for PII (optional)
                          └─ writes a JSONL audit trail (never raw content)
```

## 60-second quickstart

1. Start the proxy (requires `pip install 'multimind-sdk[gateway]'`):

```bash
export OPENAI_API_KEY=sk-...
multimind serve --port 8400 --strategy mask --block-on ssn,credit_card \
    --budget 5.00 --audit-log audit.jsonl
```

Fully local, no API key at all:

```bash
multimind serve --port 8400 --upstream ollama
```

2. Change one line in your existing app:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8400/v1", api_key="unused")

# Everything else stays exactly the same:
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize this: my SSN is 212-45-6789"}],
)
```

That request reaches the upstream as `"... my SSN is [SSN]"` (or is rejected
with a 400 if `ssn` is in `--block-on`), spend is metered against the budget,
and `audit.jsonl` records what was detected — types, counts, and hash tags
only, never the raw values.

## What it exposes

| Endpoint | Behavior |
|---|---|
| `POST /v1/chat/completions` | Full pipeline: block check, budget check, request redaction, audit, forward, output scan. Supports `stream: true` (SSE passthrough with output scanning). |
| `POST /v1/embeddings` | Block check, budget check, input redaction, audit, forward. |
| `GET /v1/models` | Forwarded to the upstream. |
| `GET /health`, `GET /ready` | Local probes; never touch the upstream. |

Error bodies follow the OpenAI `{"error": {...}}` shape:

- **400** `compliance_violation` / `pii_blocked` — a `--block-on` PII type was
  found in the request (includes `blocked_types`).
- **429** `budget_exceeded` — the session budget is spent (includes `spent`
  and `max_cost`).
- **502** `upstream_error` — the upstream could not be reached. Upstream 4xx/5xx
  responses are passed through unchanged.

## Configuration

Every option is available as a CLI flag and an environment variable; explicit
flags win.

| CLI flag | Env var | Default | Meaning |
|---|---|---|---|
| `--upstream` | `MULTIMIND_PROXY_UPSTREAM` | `openai` | Named provider: `openai`, `groq`, `mistral`, `gemini`, `deepseek`, `ollama`. API key is read from the provider's usual env var (e.g. `GROQ_API_KEY`); `ollama` uses `OLLAMA_HOST` (default `http://localhost:11434`) and needs no key. |
| `--upstream-base-url` | `MULTIMIND_UPSTREAM_BASE_URL` | — | Any OpenAI-compatible URL; overrides `--upstream`. |
| — | `MULTIMIND_UPSTREAM_API_KEY` | — | Key sent as `Authorization: Bearer ...` to a custom upstream. If unset, the client's own `Authorization` header is passed through. |
| `--strategy` | `MULTIMIND_PROXY_STRATEGY` | `mask` | Redaction strategy: `mask` (`[EMAIL]`), `hash` (`[EMAIL:1a2b3c4d]`), `remove`. |
| `--block-on` | `MULTIMIND_PROXY_BLOCK_ON` | — | Comma-separated PII types that reject the request: `email`, `phone`, `ssn`, `credit_card`, `ip_address`, `iban`, `passport`, `dob`, `api_key`. |
| `--budget` | `MULTIMIND_PROXY_BUDGET` | unlimited | Max session spend in USD; requests past the ceiling get 429. |
| — | `MULTIMIND_PROXY_COST_PER_TOKEN` | — | Blended USD/token used for budget math when the model is not in the built-in OpenAI pricing table. |
| `--audit-log` | `MULTIMIND_PROXY_AUDIT_LOG` | disabled | JSONL audit-trail path. |
| `--scan-output/--no-scan-output` | `MULTIMIND_PROXY_SCAN_OUTPUT` | on | Also redact PII in model output (including streams). |
| `--host` | `MULTIMIND_PROXY_HOST` | `127.0.0.1` | Bind host. |
| `--port` | `MULTIMIND_PROXY_PORT` | `8400` | Bind port. |

Programmatic use:

```python
from multimind.gateway.guard_proxy import ProxySettings, create_app

app = create_app(ProxySettings(upstream="ollama", block_on=("ssn",), budget=5.0))
# serve `app` with uvicorn, or mount it inside a larger FastAPI app
```

## docker-compose

The proxy is a plain uvicorn app, so it drops into the existing compose setup:

```yaml
services:
  guard-proxy:
    image: your-multimind-image
    command: multimind serve --host 0.0.0.0 --port 8400 --block-on ssn,credit_card
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MULTIMIND_PROXY_AUDIT_LOG=/data/audit.jsonl
    ports:
      - "8400:8400"
    volumes:
      - ./audit:/data
```

## Honest limitations

- **Latency**: every request takes one extra local hop plus regex scanning.
  Overhead is small (milliseconds) but not zero.
- **Streaming is slightly buffered**: to catch PII split across SSE chunks,
  the proxy holds back a 64-character overlap window before emitting each
  chunk. Output arrives marginally behind the upstream, and chunk boundaries
  are re-segmented (total content is preserved).
- **Regex-based detection**: the built-in `PIIDetector` covers common
  structured PII (emails, SSNs, cards, keys, ...). It is not an NLP entity
  recognizer and will not catch free-form PII like names or addresses.
- **Budget accuracy**: costs use real token usage when the upstream reports
  it and a chars/4 estimate otherwise (always for streams). Models outside
  the built-in OpenAI pricing table are recorded as unpriced ($0) unless you
  set `MULTIMIND_PROXY_COST_PER_TOKEN`, so the budget cannot bite there.
  The budget is per-process ("session") and resets on restart.
- **Chat surface**: string content and `{"type": "text"}` multimodal parts
  are scanned; image/audio parts pass through unscanned. Tool-call arguments
  in responses are not currently scanned.
- **Not authentication**: the proxy does not authenticate its own callers;
  bind it to localhost or put it behind your own auth in production.
