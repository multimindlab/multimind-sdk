# MultiMind Compliance

This documentation covers the compliance monitoring and evaluation features in MultiMind.

## Overview

MultiMind provides a comprehensive compliance framework that includes:
- Real-time compliance monitoring
- Healthcare-specific compliance checks
- Interactive compliance dashboard
- Alert management system
- Compliance trend analysis
- Advanced compliance mechanisms (federated shards, ZK proofs, etc.)

## Features

### Core Compliance Features
- Run compliance monitoring with custom configurations
- Generate compliance reports
- Run example compliance scenarios
- List available regulations and healthcare use cases
- Healthcare-specific compliance monitoring
- Real-time compliance evaluation
- Compliance recommendations generation

### Advanced Features
- Interactive compliance dashboard
- Alert management and monitoring
- Compliance trend analysis
- Federated compliance shards
- Zero-knowledge proofs
- Differential privacy feedback loops
- Self-healing patches
- Explainable DTOs
- Model watermarking and fingerprint tracking
- Adaptive privacy via differentially-private feedback
- Provable zero-knowledge compliance
- Dynamic regulatory change detection and auto-patch
- Federated compliance across jurisdictions

## Installation

```bash
pip install "multimind-sdk[compliance]"
```

## Usage

### Command Line Interface

The compliance features can be accessed through the MultiMind CLI:

```bash
# Scan text or a file for PII (offline, no config needed)
multimind compliance scan-text "My email is jane.doe@corp.com"

# Run compliance monitoring
multimind compliance run-compliance --config config.json --output results.json

# Run example scenarios
multimind compliance run-example --type healthcare --use-case medical_diagnosis --output results.json

# Generate compliance report
multimind compliance generate-report --config config.json --output report.json

# Show compliance dashboard
multimind compliance dashboard --organization-id org_123 --time-range 7d --use-case medical_diagnosis

# Manage alerts
multimind compliance alerts --organization-id org_123 --status active --severity high
multimind compliance configure-alerts --organization-id org_123 --config alert_rules.json
```

### API Endpoints

The compliance API is available through the MultiMind Gateway:

```bash
# Start the API server (requires the gateway extra)
pip install "multimind-sdk[gateway]"
python -m multimind.gateway.api
```

#### Available Endpoints

##### POST /v1/compliance/monitor
Run compliance monitoring with custom configuration.

Request body:
```json
{
    "organization_id": "org_123",
    "organization_name": "Example Corp",
    "dpo_email": "dpo@example.com",
    "enabled_regulations": ["HIPAA", "GDPR"],
    "compliance_rules": {
        "privacy_threshold": 0.9,
        "fairness_threshold": 0.9
    },
    "metadata": {
        "model_type": "healthcare",
        "data_categories": ["health_data"]
    }
}
```

##### GET /v1/compliance/dashboard
Get compliance dashboard metrics.

Parameters:
- `organization_id`: Organization ID
- `time_range`: Time range (e.g., 7d, 24h)
- `use_case`: Specific use case (optional)

Response:
```json
{
    "total_checks": 100,
    "passed_checks": 95,
    "failed_checks": 5,
    "compliance_score": 0.95,
    "recent_issues": [
        {
            "description": "Privacy threshold violation",
            "severity": "high"
        }
    ],
    "trend_data": {
        "compliance_score": [0.92, 0.93, 0.95],
        "privacy_score": [0.94, 0.95, 0.96],
        "fairness_score": [0.91, 0.92, 0.93],
        "transparency_score": [0.93, 0.94, 0.95]
    },
    "alerts": [
        {
            "description": "High severity compliance issue detected",
            "severity": "high"
        }
    ]
}
```

##### POST /v1/compliance/alerts/configure
Configure compliance alert rules.

Request body:
```json
{
    "organization_id": "org_123",
    "alert_rules": {
        "privacy_threshold": {
            "threshold": 0.9,
            "severity": "high",
            "notification_channels": ["email", "slack"]
        },
        "fairness_threshold": {
            "threshold": 0.9,
            "severity": "medium",
            "notification_channels": ["email"]
        }
    }
}
```

##### GET /v1/compliance/alerts
Get compliance alerts.

Parameters:
- `organization_id`: Organization ID
- `status`: Alert status (active/resolved)
- `severity`: Alert severity (high/medium/low)

## Configuration

### Compliance Rules

Default compliance rules include:
- Privacy threshold
- Fairness threshold
- Transparency threshold
- Bias threshold

Healthcare-specific rules include:
- HIPAA compliance
- Data minimization
- Audit trail
- Explainability

### Alert Rules

Alert rules can be configured for:
- Compliance threshold violations
- Privacy breaches
- Fairness issues
- Transparency concerns
- Custom compliance checks

Example alert configuration:
```json
{
    "organization_id": "org_123",
    "alert_rules": {
        "privacy_threshold": {
            "threshold": 0.9,
            "severity": "high",
            "notification_channels": ["email", "slack"],
            "cooldown_period": "1h"
        },
        "hipaa_compliance": {
            "threshold": 1.0,
            "severity": "critical",
            "notification_channels": ["email", "slack", "pagerduty"],
            "cooldown_period": "0h"
        }
    }
}
```

## Advanced Features

### Federated Compliance Shards
- Distributed compliance monitoring
- Cross-jurisdictional compliance verification
- Zero-knowledge proofs for privacy
- Self-healing compliance patches

### Adaptive Privacy
- Differential privacy feedback loops
- Privacy-preserving compliance checks
- Dynamic privacy parameter adaptation
- Privacy-aware model training

### Model Watermarking
- Unique model fingerprints
- Provenance tracking
- Compliance verification
- Model attribution

### Regulatory Change Detection
- Automated regulatory updates
- Compliance patch generation
- Cross-jurisdictional compliance mapping
- Real-time compliance adaptation

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
