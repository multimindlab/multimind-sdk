---
title: "Run Tracing"
---

`multimind.observability.tracing` records runs (root traces) and nested spans for LLM, chain, and tool calls, writes them to a local SQLite file, and optionally exports them to a self-hosted MultiMind platform over HTTP. It ships in the core install (its only dependency is `httpx`, already a core requirement).

Privacy first: by default no prompt or response text is stored or exported. Each run keeps only the character count and the SHA-256 hash of its inputs and outputs, which is enough to spot duplicates and payload drift without retaining the text.

## Quick start

```python
from multimind.observability.tracing import RunTracer, trace_model

tracer = RunTracer(project="support-bot", sqlite_path="traces.db")
model = trace_model(OpenAIModel(model_name="gpt-4o-mini"), tracer=tracer)

reply = await model.generate("Summarize this ticket ...")
```

Every `generate`, `chat`, `generate_stream`, `chat_stream`, `generate_sync`, and `chat_sync` call on the wrapped model records one `llm` run with start and end times, token usage, and cost. Any other attribute (for example `model_name`) is proxied to the wrapped model, so `trace_model` composes with `guard()` and `track_costs()`:

```python
model = trace_model(track_costs(guard(OpenAIModel()), tracker=cost_tracker), tracer=tracer)
```

## Runs and spans

`RunTracer.run()` returns a context manager; `Run.span()` opens a child. A run that is still open when its parent finishes is closed automatically, and a finished root run is dispatched to the sinks with all of its spans.

```python
with tracer.run("pipeline", inputs={"query": "hello"}) as root:
    with root.span("retrieve", "tool", inputs="hello") as step:
        step.end(outputs=["doc1", "doc2"])
    root.end(outputs="answer")

payload = tracer.completed[0]        # {"run": {...}, "spans": [{...}]}
payload["spans"][0]["parent_id"] == payload["run"]["id"]
```

`run_type` must be one of `llm`, `chain`, or `tool`; anything else raises `ValueError`. Leaving a `with` block through an exception marks the run `status="error"` and stores `"ExceptionType: message"` in `error`.

Calls made on a traced model inside an open run are nested under it automatically (ambient context via `contextvars`), so the LLM span lands beneath the chain that made it:

```python
with tracer.run("chain") as root:
    model.generate_sync("nested")
tracer.completed[-1]["spans"][0]["name"]   # "gpt-4o-mini.generate_sync"
```

The `traced` decorator records one run per call of a sync or async function:

```python
@tracer.traced("add", run_type="tool")
def add(a, b):
    return a + b

@tracer.traced
async def fetch(x):
    ...
```

## What a run contains

Each run and span is a dict with the same fields, which is also the row layout of the SQLite `runs` table and the body sent to the platform:

| Field | Meaning |
| --- | --- |
| `id`, `parent_id` | UUIDs; `parent_id` is `null` on the root run |
| `project`, `name`, `run_type`, `tags` | `project` and `tags` come from the tracer; per-run tags are appended |
| `start_time`, `end_time` | ISO-8601 UTC |
| `status`, `error` | `success` or `error`, plus the error text |
| `model`, `provider` | Taken from the wrapped model's `model_name` and `PROVIDER_NAME` for traced models |
| `input_chars`, `output_chars` | Lengths of the canonical text form (JSON for non-string inputs) |
| `input_sha256`, `output_sha256` | SHA-256 of that text |
| `inputs`, `outputs` | Always `null` unless `store_content=True` |
| `tokens` | `{"input", "output", "estimated"}`; `estimated` is `true` when the model exposed no usage and the chars/4 estimate was used |
| `cost` | USD from the model's `cost_per_token` or class `MODEL_PRICING`; `null` when the model is unpriced |

## Privacy defaults and `store_content`

The default tracer never writes prompt or completion text anywhere: `inputs` and `outputs` are `null` in memory, in SQLite, and in the export payload. The tests in `tests/test_tracing.py` assert that a prompt containing an SSN appears neither in the HTTP body nor in the SQLite file bytes.

Pass `store_content=True` to keep inputs and outputs verbatim. This makes traces far easier to debug, but it means prompts and completions, including any PII they contain, are written to the SQLite file and sent to the platform. Pair it with the [PII guard](/compliance) if you need it in a regulated environment.

```python
tracer = RunTracer(sqlite_path="traces.db", store_content=True)
```

## SQLite sink

`sqlite_path` creates the file (and parent directories) on construction with a single `runs` table whose columns are the fields above; `tokens` and `tags` are stored as JSON text. Root runs and spans are rows in the same table linked by `parent_id`.

```python
import sqlite3
rows = sqlite3.connect("traces.db").execute(
    "SELECT name, run_type, status, cost FROM runs WHERE parent_id IS NULL"
).fetchall()
```

Write failures are logged as warnings; tracing never raises into the host application.

## HTTP export

```python
tracer = RunTracer(
    project="support-bot",
    export_url="https://platform.example.com",
    api_key="...",
)
```

Each completed root run is sent as `POST {export_url}/api/v1/runs` with an `X-API-Key` header and the JSON body `{"run": {...}, "spans": [{...}, ...]}`. A failed request (connection error or HTTP 4xx/5xx) is logged and the payload stays buffered; `tracer.pending_exports` reports how many are waiting and `tracer.flush()` retries them. `flush()` is also registered with `atexit` so buffered runs are retried at interpreter shutdown.

## Default tracer and environment variables

`trace_model(model)` without an explicit `tracer` uses a process-wide tracer built from the environment on first use:

| Variable | Tracer argument |
| --- | --- |
| `MULTIMIND_TRACING_PROJECT` | `project` (default `default`) |
| `MULTIMIND_TRACING_DB` | `sqlite_path` |
| `MULTIMIND_PLATFORM_URL` | `export_url` |
| `MULTIMIND_PLATFORM_KEY` | `api_key` |

`get_default_tracer()` returns that instance and `reset_default_tracer()` discards it (useful in tests after changing the environment). `store_content` cannot be enabled through the environment; it must be passed explicitly.

## API summary

| Name | Purpose |
| --- | --- |
| `RunTracer(project, export_url, api_key, sqlite_path, store_content, tags)` | Owns the sinks; `start_run()`, `run()`, `traced()`, `flush()`, `pending_exports`, `completed` |
| `Run` | One traced unit; `span()`, `end(outputs, error, tokens, cost)`, `to_dict()`, usable as a context manager |
| `TracedModel(model, tracer, name)` | Model wrapper; `name` overrides the `model_name` prefix used in run names |
| `trace_model(model, tracer=None, name=None)` | Convenience constructor for `TracedModel` on the default tracer |
| `get_default_tracer()` / `reset_default_tracer()` | Process-wide tracer from `MULTIMIND_*` variables |

Source: [multimind/observability/tracing.py](https://github.com/multimindlab/multimind-sdk/blob/develop/multimind/observability/tracing.py). Tests: [tests/test_tracing.py](https://github.com/multimindlab/multimind-sdk/blob/develop/tests/test_tracing.py).
