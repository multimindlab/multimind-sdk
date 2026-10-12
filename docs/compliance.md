---
title: "MultiMind Compliance"
---

This documentation covers the compliance monitoring and evaluation features in MultiMind.

## Overview

MultiMind's compliance package (`multimind.compliance`) has three layers:

- **Runtime guard** (stdlib-only, no extras): `PIIDetector`, `ComplianceGuard`
  / `guard()`, `AuditLog`, and `ComplianceViolationError`. These power
  `multimind serve` ([guard-proxy.md](/guard-proxy)), the framework
  adapters ([integrations.md](/integrations)), and the MCP server
  ([mcp-server.md](/mcp-server)).
- **Evidence reporting** (stdlib-only): `build_evidence_report` /
  `EvidenceReport`, and `RegulatoryWatcher` (content-hash change detection
  over regulatory web pages).
- **Regulation managers** (need the `compliance` extra): `GovernanceConfig`,
  `Regulation`, `PrivacyCompliance`, `GDPRCompliance`, and the per-domain
  managers in `multimind.compliance.ai_frameworks`, `data_transfer`,
  `accessibility`, `supply_chain`, and `corporate`. See the
  [quickstart](/compliance_quickstart) for working examples.

The compliance monitoring CLI/API described below wraps the runtime guard and
these managers. The "advanced" classes also exported from
`multimind.compliance` (`FederatedCompliance`, `ZeroKnowledgeProof`,
`ModelWatermarking`, `SelfHealingCompliance`, `ExplainableDTO`,
`RegulatoryChangeDetector`) are fail-closed stubs in 1.0.0 — see
[Advanced features](#advanced-features) below before relying on any of them.

## Installation

```bash
pip install "multimind-sdk[compliance]"
```

`scan-text` and `report-evidence` work on a core install. The monitoring
commands (`run-compliance`, `generate-report`, `dashboard`, `alerts`,
`configure-alerts`) import the gateway's compliance API and need both
extras: `pip install "multimind-sdk[compliance,gateway]"`.

## Usage

### Command Line Interface

The compliance features can be accessed through the MultiMind CLI
(`multimind compliance --help` lists every subcommand):

```bash
# Scan text or a file for PII (offline, no config needed); exit code 1 if PII found
multimind compliance scan-text "My email is jane.doe@corp.com"
multimind compliance scan-text --file prompt.txt --redact mask

# Run compliance monitoring (config format: see POST /v1/compliance/monitor below)
multimind compliance run-compliance --config config.json --output results.json

# Generate compliance report
multimind compliance generate-report --config config.json --output report.json

# Show compliance dashboard
multimind compliance dashboard --organization-id org_123 --time-range 7d --use-case medical_diagnosis

# Manage alerts
multimind compliance alerts --organization-id org_123 --status active --severity high
multimind compliance configure-alerts --organization-id org_123 --config alert_rules.json

# Run the bundled example scenarios (needs a source checkout: imports from examples/compliance/)
multimind compliance run-example --type healthcare --use-case medical_diagnosis --output results.json
```

Known limitations in 1.0.0:

- `dashboard`, `alerts`, and `configure-alerts` read an in-memory store that
  is filled by `run-compliance` calls made in the same process. From the CLI
  each command is its own process, so these three print nothing recorded.
  Use the `/v1/compliance/*` routes on a running `multimind backend`, where
  every `POST /v1/compliance/monitor` is recorded until the server restarts.
  The durable artifacts are the JSONL audit/cost logs consumed by
  `multimind dashboard` ([dashboard.md](/dashboard)) and `report-evidence` below.
- `run-example` imports `examples.compliance.*` from the repository root, so
  it only works from a git checkout, not from a pip-installed package, and
  the example scripts import `torch` (`[finetune]` extra). The error message
  names which of the two is missing.

### API Endpoints

The compliance API is a router mounted at `/v1/compliance` on the MultiMind
Gateway (`multimind/gateway/compliance_api.py`):

```bash
# Start the API server (requires the gateway extra); listens on 0.0.0.0:8000
pip install "multimind-sdk[compliance,gateway]"
python -m multimind.gateway.api
```

Swagger UI is at `/docs`. The same router is also reachable under
`/gateway/v1/compliance/...` when running `multimind backend`.

#### Available Endpoints

##### POST /v1/compliance/monitor
Run compliance monitoring against the implemented checkers. Missing
evidence is reported as `unverifiable`, never as a pass.

Request body (`ComplianceConfig`; all fields except `metadata` are required):
```json
{
    "organization_id": "org_123",
    "organization_name": "Example Corp",
    "dpo_email": "dpo@example.com",
    "enabled_regulations": ["HIPAA", "GDPR"],
    "compliance_rules": {
        "data_retention_days": 90
    },
    "metadata": {
        "text": "Prompt or payload to scan for PII",
        "purpose": "analytics",
        "data_categories": ["health_data"]
    }
}
```

`enabled_regulations` must be `Regulation` enum names (`GDPR`, `AI_ACT`,
`HIPAA`, `SOC2`, ... — `GET /v1/compliance/regulations` lists them);
unknown names return 400. What is actually evaluated:

- `metadata.text` (if present) is run through `PIIDetector` → a `pii_scan`
  check with the detected types and a `redact_or_block` recommendation.
- `GDPR` with `metadata.purpose` is evaluated with `GDPRPolicy`, using
  `compliance_rules.data_retention_days` (default 365).
- Every other enabled regulation yields an `unverifiable` check. Threshold
  keys such as `privacy_threshold` are accepted but not evaluated.

Response:
```json
{
    "final_evaluation": {
        "status": "attention",
        "checks": [
            {"check": "pii_scan", "status": "findings", "types": ["email"], "count": 1},
            {"regulation": "HIPAA", "status": "unverifiable", "detail": "No runtime evidence was supplied for HIPAA. A missing checker is not treated as compliant."}
        ]
    },
    "recommendations": [{"action": "redact_or_block", "detail": "PII types detected: email"}],
    "metrics": {}
}
```

`status` is `checked` (all checks clear), `attention` (findings or denied
checks), or `unverifiable` (no checks, or any check unverifiable).

##### POST /v1/compliance/report
Same body as `/monitor`; returns the monitoring result plus a per-regulation
`enabled`/`disabled` map. It does not compute a compliance score.

##### GET /v1/compliance/regulations, GET /v1/compliance/healthcare/use-cases
List the `Regulation` enum names and the healthcare use-case names.

##### GET /v1/compliance/dashboard
Parameters: `organization_id` (required), `time_range` (`7d`, `24h`, ...),
`use_case` (optional).

Returns a `DashboardMetrics` object: `total_checks`, `passed_checks`,
`failed_checks`, `unverifiable_checks`, `compliance_score`
(passed divided by passed plus failed), `recent_issues` (last five failed
checks with `description` and `severity`), `trend_data` with one
`compliance_score` series of 1.0/0.0 per decided check, and `alerts`.
Every `POST /v1/compliance/monitor` on this server process is recorded under
its `organization_id`: a `checked` result counts as passed, `attention` as
failed and raises an active alert, `unverifiable` is counted separately.
No privacy, fairness, or transparency score is invented. The store is
in-memory and empties when the server restarts.

##### POST /v1/compliance/alerts/configure
Query parameter `organization_id`; body is the `alert_rules` dict. Stored
on the organization's in-memory shard; `{"enabled": false}` stops new
alerts from being raised. No other rule keys are evaluated.

##### GET /v1/compliance/alerts
Parameters: `organization_id` (required), `status` (default `active`),
`severity` (optional). Always returns `[]` in 1.0.0 for the same reason.

## Configuration

### Compliance Rules

`compliance_rules` is a free-form dict on the monitor/report request. The
only key the 1.0.0 checkers read is `data_retention_days` (GDPR). Model
training thresholds (`bias_threshold`, `privacy_threshold`, ...) belong to
`multimind.compliance.model_training.ComplianceTrainer`, documented in the
[quickstart](/compliance_quickstart#model-training-with-compliance).

### Alert Rules

`configure-alerts` / `POST /v1/compliance/alerts/configure` accept any JSON
object as `alert_rules`; no schema is enforced and no notification channel
(email, Slack, PagerDuty) is implemented. Treat alert rules as not yet
functional.

## Advanced features

The classes exported from `multimind.compliance.advanced` are deliberately
fail-closed: rather than fabricate a proof, patch, watermark, or
explanation, they raise `NotImplementedError` until a real backend is wired
in. Status in 1.0.0 (verified by calling each one):

| Class | Status |
|---|---|
| `ZeroKnowledgeProof`, `HomomorphicEncryption` | `prove`/`verify`/`encrypt` raise `NotImplementedError`. |
| `FederatedCompliance` | `verify_global_compliance` always raises (no coordinator/consensus backend). |
| `ComplianceShard` | `verify_compliance` raises (`_apply_local_rules` not implemented). Its in-memory history and alert lists are filled only by the gateway's `/v1/compliance/monitor`. |
| `RegulatoryChangeDetector` | Constructor fails with `AttributeError` (`_initialize_regulatory_sources` is missing). Use `RegulatoryWatcher` instead. |
| `SelfHealingCompliance` | `check_and_heal` returns the state unchanged when nothing is detected; any detected vulnerability raises (patch application not implemented). |
| `ExplainableDTO` | `explain_decision` raises (no explanation backend). |
| `ModelWatermarking` | `watermark_model` and `verify_watermark` raise; `track_fingerprint` works (SHA-256 of the model's type and `repr`). |
| `AdaptivePrivacy`, `DPMechanism`, `PrivacyBudget` | Work: epsilon adaptation from `compliance_score`/`loss` feedback and Laplace noise for numeric data. |

What does work for regulatory change detection is `RegulatoryWatcher`:
hash-based monitoring of regulatory web pages with a raw unified diff and no
legal interpretation.

```python
from multimind.compliance import RegulatoryWatcher

watcher = RegulatoryWatcher(
    sources=[{"name": "eu-ai-act", "url": "https://example.org/ai-act"}],
    state_path="regwatch_state.json",
)
events = await watcher.check()            # fetches each source over HTTP
# events: list[ChangeEvent] with name, url, changed, diff, checked_at, previous_hash, current_hash
# watcher.check_offline({"eu-ai-act": "local_copy.html"}) does the same from local files
```

## Evidence reports

`multimind compliance report-evidence` turns the SDK's JSONL artifacts — the
`ComplianceGuard` audit trail, the `CostTracker` cost log, and an optional AI
inventory scan — into a single auditor-shaped evidence document (Markdown or
self-contained HTML). It works on core installs (stdlib-only, no extras) and
is deterministic given the same inputs.

```bash
multimind compliance report-evidence \
  --audit-log audit.jsonl \
  --costs-log costs.jsonl \
  --project . \
  --period 2026-07 \
  --format html -o evidence.html
```

Report sections:

1. Sources (which artifacts were provided; absent sources are stated, never inferred)
2. Data protection (PII events by type, redaction strategy distribution, blocked requests, scanned-call share)
3. Oversight and audit-trail continuity (first/last record, gaps over a configurable threshold, records by direction/method)
4. Cost governance (spend by tag and model, budget-block events, estimated/unpriced call counts)
5. AI asset inventory (providers in use, external data-flow providers, hardcoded-key findings)
6. Framework control-theme mapping (EU AI Act Art. 12 record-keeping / Art. 50 transparency, SOC 2 monitoring / logical access, HIPAA 164.312(b) audit controls — phrased as "supports", never "satisfies")
7. Disclaimer

What it is NOT: the report is technical evidence generated from runtime
artifacts, not legal advice, a certification, or a compliance determination.
It documents only what the artifacts record; an absent artifact means absent
evidence, not absent risk. Every report carries this disclaimer.

The same report is available programmatically:

```python
from multimind.compliance import build_evidence_report

report = build_evidence_report(audit_log="audit.jsonl", costs_log="costs.jsonl", period="2026-07")
print(report.to_markdown())  # or report.to_html(), report.to_dict()
```

## Development

### Running Tests

```bash
pytest tests/compliance/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the Apache License - see the LICENSE file for details. 
