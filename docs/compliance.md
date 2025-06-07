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
pip install multimind[compliance]
```

## Usage

### Command Line Interface

The compliance features can be accessed through the MultiMind CLI:

```bash
# Run compliance monitoring
multimind compliance run --config config.json --output results.json

# Run example scenarios
multimind compliance example --type healthcare --use-case medical_diagnosis --output results.json

# Generate compliance report
multimind compliance report --config config.json --output report.json

# Show compliance dashboard
multimind compliance dashboard --organization-id org_123 --time-range 7d --use-case medical_diagnosis

# Manage alerts
multimind compliance alerts --organization-id org_123 --status active --severity high
multimind compliance configure-alerts --organization-id org_123 --config alert_rules.json
```

### API Endpoints

The compliance API is available through the MultiMind Gateway:

```bash
# Start the API server
multimind gateway start
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

## Development

### Running Tests

```bash
pytest tests/compliance/
```

### Building Documentation

```bash
cd docs
make html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the Apache License - see the LICENSE file for details. 
