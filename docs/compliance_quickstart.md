---
title: "Compliance Framework Quickstart Guide"
---

This guide provides a quick introduction to using the MultiMind Compliance Framework. For detailed documentation, see the [Compliance Framework Guide](/compliance).

## Installation

The regulation managers used below (`GovernanceConfig`, `PrivacyCompliance`,
the domain managers, and the advanced classes) are gated behind the
`compliance` extra; the runtime PII guard and evidence reporting work on a
core install. Install with pip (Python 3.10+):

```bash
pip install "multimind-sdk[compliance]"
```

## Basic Setup

1. Import the required modules:

```python
from multimind.compliance import GovernanceConfig, Regulation, PrivacyCompliance
from multimind.compliance.ai_frameworks import AIFrameworkCompliance
from multimind.compliance.data_transfer import DataTransferCompliance
from multimind.compliance.accessibility import AccessibilityCompliance
from multimind.compliance.supply_chain import SupplyChainCompliance
from multimind.compliance.corporate import CorporateCompliance
```

2. Configure the governance settings:

```python
config = GovernanceConfig(
    organization_id="org_123",
    organization_name="Your Organization",
    dpo_email="dpo@yourorg.com",
    enabled_regulations=[
        Regulation.GDPR,
        Regulation.AI_ACT,
        Regulation.HIPAA
    ]
)
```

## Common Use Cases

### 1. Privacy Compliance

```python
# Initialize privacy compliance
privacy = PrivacyCompliance(config=config)

# Process a data subject access request
async def handle_dsar(user_id: str):
    result = await privacy.process_data_subject_request(
        request_type="access",
        user_id=user_id,
        data_ids=["data_123"]
    )
    return result

# Check retention compliance
async def check_retention():
    issues = await privacy.check_retention_compliance()
    return issues
```

### 2. AI System Compliance

```python
# Initialize AI compliance
ai_compliance = AIFrameworkCompliance(config=config)

# Assess AI system compliance
async def assess_ai_system(system_id: str):
    result = await ai_compliance.assess_oecd_compliance(
        system_id=system_id,
        system_metadata={"type": "classification"}
    )
    return result
```

### 3. Cross-Border Data Transfer

```python
# Initialize data transfer compliance
transfer = DataTransferCompliance(config=config)

# Validate international data transfer
async def validate_transfer(source_country: str, destination_country: str):
    result = await transfer.validate_schrems_ii_compliance(
        transfer_id="transfer_123",
        source_country=source_country,
        destination_country=destination_country,
        data_categories=["personal_data"],
        transfer_mechanism="SCC"
    )
    return result
```

### 4. Accessibility Compliance

```python
# Initialize accessibility compliance
accessibility = AccessibilityCompliance(config=config)

# Validate WCAG compliance
async def validate_accessibility(system_id: str):
    result = await accessibility.validate_wcag_compliance(
        assessment_id="wcag_123",
        system_id=system_id,
        version="2.1"
    )
    return result
```

### 5. Supply Chain Compliance

```python
# Initialize supply chain compliance
supply_chain = SupplyChainCompliance(config=config)

# Assess vendor security
async def assess_vendor(vendor_id: str):
    result = await supply_chain.assess_vendor_security(
        vendor_id=vendor_id,
        vendor_name="Vendor Name",
        assessment_type="SIG"
    )
    return result
```

### 6. Corporate Compliance

```python
# Initialize corporate compliance
corporate = CorporateCompliance(config=config)

# Assess SOX compliance
async def assess_sox(system_id: str):
    result = await corporate.assess_sox_compliance(
        assessment_id="sox_123",
        system_id=system_id,
        fiscal_year="2024"
    )
    return result
```

## Advanced Compliance Features

The classes in `multimind.compliance.advanced` (re-exported from
`multimind.compliance`) each take a plain `dict` configuration, but most of
them are **fail-closed stubs in 1.0.0**: rather than fabricate a proof,
patch, watermark, or explanation, they raise `NotImplementedError` until a
real backend is integrated. The status below was verified by calling each
method; see [compliance.md](/compliance#advanced-features) for the full
table.

### 1. Federated Compliance — unavailable

`FederatedCompliance(config).verify_global_compliance(data)` always raises
`NotImplementedError` (no coordinator or consensus backend ships), and
`ComplianceShard.verify_compliance` raises because local rule evaluation is
not implemented. Do not build on these yet.

### 2. Regulatory Change Detection — use `RegulatoryWatcher`

`RegulatoryChangeDetector(config={})` fails on construction in 1.0.0
(`AttributeError: '_initialize_regulatory_sources'`), and its patch
validation is unimplemented. The working alternative is the hash-based
watcher, which needs no extras and makes no legal interpretation:

```python
from multimind.compliance import RegulatoryWatcher

watcher = RegulatoryWatcher(
    sources=[{"name": "eu-ai-act", "url": "https://example.org/ai-act"}],
    state_path="regwatch_state.json",   # persists last-seen hashes between runs
)
events = await watcher.check()        # HTTP fetch of each source
for e in events:                      # ChangeEvent: name, url, changed, diff, checked_at, previous_hash, current_hash
    if e.changed:
        print(e.name, e.diff)

# Dry run from local copies instead of the network:
events = watcher.check_offline({"eu-ai-act": "local_copy.html"})
```

### 3. Zero-Knowledge Compliance Proofs — unavailable

`multimind.compliance.advanced.ZeroKnowledgeProof` is a fail-closed stub: without a real ZKP backend installed, `prove` and `verify` raise `NotImplementedError` rather than fabricating a proof. `HomomorphicEncryption.encrypt` likewise raises. Treat these as unavailable until a backend integration ships.

### 4. Differential Privacy — works

```python
from multimind.compliance import AdaptivePrivacy
from multimind.compliance.advanced import DPMechanism

# Adapt epsilon from feedback. Recognised config keys: initial_epsilon (1.0),
# min_epsilon (0.1), max_epsilon (10.0), adaptation_rate (0.1), dp_sensitivity (1.0).
privacy_loop = AdaptivePrivacy(config={"initial_epsilon": 1.0})
# Recognised feedback keys: compliance_score (drives adaptation) and loss.
await privacy_loop.adapt_privacy(feedback={"compliance_score": 0.65})
print(privacy_loop.dp_mechanism.epsilon)   # 0.2: stepped up by adaptation_rate because compliance < 0.7
# Caveat: the walk starts from the internal HomomorphicEncryption stub's
# epsilon (0.1), not from initial_epsilon, so the first step lands near
# min_epsilon regardless of the configured starting value.

# Laplace noise for numeric values / lists of numbers (strings and bools raise NotImplementedError)
noisy = DPMechanism(epsilon=1.0, sensitivity=1.0).privatize(42.0)
```

### 5. Model Fingerprinting — partial

```python
from multimind.compliance import ModelWatermarking

watermarking = ModelWatermarking(config={})
info = await watermarking.track_fingerprint(model)
# {"fingerprint": "fingerprint_<sha256>", "timestamp": ..., "model_id": ...}
# The fingerprint hashes the model's type name and repr(); it identifies the
# object, not the weights.
```

`watermark_model` and `verify_watermark` raise `NotImplementedError` (no
watermarking backend), so there is no actual watermark embedding or
verification in 1.0.0.

### 6. Self-Healing Policies — unavailable

`SelfHealingCompliance(config).check_and_heal(state)` returns the state
unchanged (with `patches_applied: []`) when it detects nothing, and raises
`NotImplementedError` as soon as a vulnerability is detected (for example
`{"status": "needs_healing"}`), because automatic patch application is not
implemented.

### 7. Explainable Compliance — unavailable

`ExplainableDTO(config).explain_decision(decision)` raises
`NotImplementedError` (no explanation model backend is configured).

## Model Training with Compliance

`multimind.compliance.model_training` wraps a PyTorch training loop with
compliance metrics. It needs `torch` (not part of the `compliance` extra;
install `multimind-sdk[finetune]` or `torch` directly): `train()` iterates
`DataLoader` batches and expects each batch to be a dict with `"input"` and
`"target"` keys. The bias/privacy/transparency/fairness scores are
heuristic and the trainer never stops training on a violation
(`_should_stop_training` always returns `False`).

### 1. Basic Setup

```python
from multimind.compliance.model_training import (
    ComplianceDataset,
    ComplianceTrainer,
    ComplianceMetrics
)

# Initialize compliance trainer
compliance_rules = {
    "bias_threshold": 0.1,
    "privacy_threshold": 0.8,
    "transparency_threshold": 0.8,
    "fairness_threshold": 0.8
}
trainer = ComplianceTrainer(
    model=your_model,
    compliance_rules=compliance_rules,
    training_config={
        "epochs": 10,
        "thresholds": compliance_rules,
        "evaluation_metrics": [
            "bias",
            "privacy",
            "transparency",
            "fairness"
        ]
    }
)
```

### 2. Dataset Compliance

```python
# Wrap your dataset with compliance checks
compliance_dataset = ComplianceDataset(
    base_dataset=your_dataset,
    compliance_rules={
        "privacy_threshold": 0.8,
        "fairness_threshold": 0.8,
        "transparency_threshold": 0.8
    },
    data_categories=["personal_data", "health_data"]
)
```

### 3. Training with Monitoring

```python
# Train model with compliance monitoring
results = await trainer.train(
    train_data=train_loader,
    val_data=val_loader,
    metadata={
        "model_type": "classification",
        "data_categories": ["personal_data", "health_data"],
        "jurisdiction": "EU"
    }
)
```

### 4. Compliance Evaluation

```python
# Get compliance evaluation results
evaluation = results["final_evaluation"]
print("Compliance Scores:", evaluation["compliance_scores"])
print("Violations:", evaluation["violations"])
print("Recommendations:", evaluation["recommendations"])
```

### 5. Saving Results

```python
# Save training results and compliance documentation
trainer.save_training_results(
    results=results,
    path="training_results.json"
)
```

## Best Practices

1. **Start with Core Regulations**
   - Begin with GDPR and AI Act
   - Add more regulations as needed
   - Keep configurations up to date

2. **Regular Assessments**
   - Schedule regular compliance checks
   - Monitor for violations
   - Document all assessments

3. **Error Handling**
   - Implement proper error handling
   - Log compliance violations
   - Set up alerts for critical issues

4. **Documentation**
   - Keep records of all assessments
   - Document compliance decisions
   - Maintain audit trails

## Best Practices for Advanced Features

1. **Regulatory Monitoring** (`RegulatoryWatcher`)
   - Keep `state_path` on durable storage so diffs span runs
   - Review raw diffs yourself; the watcher does not interpret them

2. **Differential Privacy** (`AdaptivePrivacy`, `DPMechanism`, `PrivacyBudget`)
   - Choose epsilon deliberately and track spend with `PrivacyBudget`
   - Validate that the noise scale fits your data's sensitivity

3. **Model Fingerprinting** (`ModelWatermarking.track_fingerprint`)
   - Remember the fingerprint is derived from `repr(model)`, not weights
   - Store fingerprint records alongside your audit trail

The federated, zero-knowledge, self-healing, and explainable-DTO classes are
fail-closed stubs in 1.0.0 (see above); there is nothing to configure for
them yet.

## Best Practices for Model Training

1. **Data Preparation**
   - Ensure data meets privacy requirements
   - Check for bias in training data
   - Document data sources and processing

2. **Compliance Monitoring**
   - Set appropriate thresholds
   - Monitor metrics during training
   - Handle violations promptly

3. **Evaluation**
   - Use comprehensive metrics
   - Test across different scenarios
   - Document evaluation results

4. **Documentation**
   - Keep detailed training logs
   - Document compliance decisions
   - Maintain audit trails

## Next Steps

1. Review the [Compliance Framework Guide](/compliance) for detailed documentation
2. Explore specific compliance modules based on your needs
3. Set up monitoring and alerting
4. Implement regular compliance checks

## Support

For help:
- Check the [Compliance Framework Guide](/compliance)
- Open a GitHub issue
- Contact the development team
- Join the community forum 