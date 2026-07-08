# Compliance Framework Quickstart Guide

This guide provides a quick introduction to using the MultiMind Compliance Framework. For detailed documentation, see the [Compliance Framework Guide](compliance.md).

## Installation

The compliance framework is included in the MultiMind SDK. Install it using pip:

```bash
pip install multimind-sdk
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

The MultiMind SDK includes cutting-edge compliance features that set it apart from other frameworks. Here's how to use them:

These classes live in `multimind.compliance` and each takes a plain `dict` configuration.

### 1. Federated Compliance

```python
from multimind.compliance import FederatedCompliance

# Verify compliance across jurisdiction-specific shards
federated = FederatedCompliance(config={"jurisdictions": ["EU", "US"]})
result = await federated.verify_global_compliance(
    data={"data_categories": ["personal_data", "health_data"], "operation": "process"}
)
```

### 2. Regulatory Change Detection

```python
from multimind.compliance import RegulatoryChangeDetector

detector = RegulatoryChangeDetector(config={})
changes = await detector.detect_changes()
patches = await detector.generate_patches(changes)
```

### 3. Zero-Knowledge Compliance Proofs

`multimind.compliance.advanced.ZeroKnowledgeProof` is currently a fail-closed stub: without a real ZKP backend installed, `prove` and `verify` raise `NotImplementedError` rather than fabricating a proof. Treat this feature as unavailable until a backend integration ships.

### 4. Differential Privacy

```python
from multimind.compliance import AdaptivePrivacy

# Adapt privacy parameters from usage feedback
privacy_loop = AdaptivePrivacy(config={"epsilon": 1.0})
await privacy_loop.adapt_privacy(
    feedback={"document_views": 100, "search_queries": 50}
)
```

### 5. Model Watermarking and Fingerprinting

```python
from multimind.compliance import ModelWatermarking

watermarking = ModelWatermarking(config={})
model = await watermarking.watermark_model(model)
fingerprint = await watermarking.track_fingerprint(model)
verification = await watermarking.verify_watermark(model)
```

### 6. Self-Healing Policies

```python
from multimind.compliance import SelfHealingCompliance

healer = SelfHealingCompliance(config={})
result = await healer.check_and_heal(
    compliance_state={"type": "data_leak", "severity": "high"}
)
```

### 7. Explainable Compliance

```python
from multimind.compliance import ExplainableDTO

explainer = ExplainableDTO(config={})
dto = await explainer.explain_decision(
    decision={
        "response_id": "resp_123",
        "rules_applied": ["gdpr.data_minimization", "eu_ai_act.transparency"],
    }
)
```

## Model Training with Compliance

The MultiMind SDK provides tools for training models while ensuring regulatory compliance. Here's how to use them:

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

1. **Federated Compliance**
   - Keep policy shards up to date
   - Monitor jurisdiction changes
   - Test with different locales

2. **Regulatory Monitoring**
   - Configure appropriate sources
   - Set up change notifications
   - Review changes regularly

3. **Zero-Knowledge Proofs**
   - Use appropriate proof types
   - Maintain verification keys
   - Document proof generation

4. **Differential Privacy**
   - Choose appropriate epsilon values
   - Monitor privacy budget
   - Validate noise addition

5. **Model Fingerprinting**
   - Generate fingerprints consistently
   - Store fingerprint data securely
   - Use for audit trails

6. **Self-Healing Policies**
   - Define clear violation thresholds
   - Set up notification channels
   - Test rollback procedures

7. **Compliance DTOs**
   - Include relevant metadata
   - Maintain audit trails
   - Use for transparency

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

1. Review the [Compliance Framework Guide](compliance.md) for detailed documentation
2. Explore specific compliance modules based on your needs
3. Set up monitoring and alerting
4. Implement regular compliance checks

## Support

For help:
- Check the [Compliance Framework Guide](compliance.md)
- Open a GitHub issue
- Contact the development team
- Join the community forum 