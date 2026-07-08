# Shadow-AI Inventory & Cost Chargeback

Static discovery of AI usage across a codebase, plus per-tag cost
attribution for chargeback reports. Both features live in
`multimind.observability` and are exposed on the `multimind audit` CLI.

## What the scanner finds

`scan_project(path)` walks a directory tree (skipping `.git`,
`node_modules`, `venv`/`.venv`, `__pycache__`, `dist`, `build`, and
similar) and reports:

| Finding kind    | Source                                                       | Example evidence        |
| --------------- | ------------------------------------------------------------ | ----------------------- |
| `dependency`    | `requirements*.txt`, `pyproject.toml`, `package.json`, `Pipfile` matched against a curated AI package registry | `openai`, `langchain-core`, `@anthropic-ai/sdk` |
| `api_url`       | AI API base URLs in `*.py` / `*.js` / `*.ts` source          | `api.openai.com`        |
| `env_read`      | Reads of known AI credential env-var names in source         | `OPENAI_API_KEY`        |
| `hardcoded_key` | Hardcoded AI API keys (PIIDetector `api_key` patterns)       | `api_key (openai)`      |
| `env_set`       | (only with `include_env=True`) known AI env keys set in the current environment | `ANTHROPIC_API_KEY` |

The package registry (`AI_PACKAGE_REGISTRY`) maps each package to a
provider, a kind (`sdk` or `framework`), and a `data_flow` classification
(`external` — data leaves your infrastructure — or `local`, e.g.
`transformers`, `ollama`). Prefix families cover `langchain*`,
`llama-index*`, `azure-ai-*`, `@langchain/*`, and `@ai-sdk/*`.

## What it never records

- **Key values.** Hardcoded-key findings record the key TYPE, file, and
  line only. The matched secret never appears in findings, summaries,
  JSON output, or CLI output.
- **Env-var values.** Environment findings record variable names only.
- **File contents.** Only classification evidence (package name, URL
  host, env-var name, key type) is retained.

Scans are honest about coverage: files skipped due to the size cap
(default 1 MB), the file-count cap (default 5000), or read errors are
counted and surfaced as a `skipped N file(s)` note — never silently
dropped.

## Quickstart

```python
from multimind.observability import scan_project

report = scan_project("path/to/project", include_env=False)
print(report.summary())   # counts by provider / kind / data_flow
print(report.risks())     # external-data-flow providers + hardcoded keys
data = report.to_dict()   # JSON-serializable
```

CLI:

```bash
multimind audit scan .            # rich table + risk summary
multimind audit scan . --json     # machine-readable output
multimind audit scan . --include-env
```

`audit scan` exits with code `1` when hardcoded AI API keys are found,
`0` otherwise — suitable for CI gates.

## JSON output shape

```json
{
  "root": "/path/to/project",
  "findings": [
    {"kind": "dependency", "provider": "openai", "evidence": "openai",
     "file": "requirements.txt", "line": 1, "data_flow": "external"},
    {"kind": "hardcoded_key", "provider": "openai", "evidence": "api_key (openai)",
     "file": "app.py", "line": 2, "data_flow": "external"}
  ],
  "summary": {
    "findings": 2,
    "by_provider": {"openai": 2},
    "by_kind": {"dependency": 1, "hardcoded_key": 1},
    "by_data_flow": {"external": 2},
    "scanned_files": 2,
    "skipped_files": 0
  },
  "risks": {
    "external_data_flow_providers": ["openai"],
    "hardcoded_keys": [{"kind": "hardcoded_key", "provider": "openai",
                        "evidence": "api_key (openai)", "file": "app.py",
                        "line": 2, "data_flow": "external"}]
  }
}
```

A top-level `"note"` field appears when files were skipped.

## Cost chargeback

`CostTracker` (see `multimind.observability.cost_tracker`) already records
per-call token/cost entries with optional tags. Chargeback groups those
records per tag:

```python
from multimind.observability import CostTracker, load_tracker, track_costs

tracker = CostTracker(jsonl_path="costs.jsonl")
model = track_costs(my_model, tracker=tracker, tag="team-a")
# ... make calls ...

data = tracker.chargeback()          # dict: per-tag cost/tokens/calls/share_pct
print(tracker.report_chargeback())   # plain-text table

# Offline, over a persisted log:
historical = load_tracker("costs.jsonl")
print(historical.report_chargeback(period="2026-07"))  # ISO timestamp prefix
```

Untagged calls are grouped under `(untagged)`. Each tag reports `calls`,
`input_tokens`/`output_tokens`/`total_tokens`, `cost`, `estimated_calls`,
`unpriced_calls`, and `share_pct` (percentage of total cost; shares sum
to 100 when any cost was recorded).

CLI:

```bash
multimind audit costs                      # session tracker
multimind audit costs --log costs.jsonl    # historical JSONL log
multimind audit costs --log costs.jsonl --period 2026-07
```
