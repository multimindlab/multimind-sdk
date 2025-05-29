# MultiMind CLI Examples

## Compliance and Governance CLI

The `compliance_cli.py` provides a command-line interface for managing compliance and governance features. Here are some example commands:

### Dataset Ingestion

```bash
# Ingest a new dataset with compliance checks
multimind governance ingest \
    --dataset-id "customer_support_2024" \
    --name "Customer Support Dataset" \
    --description "Customer support conversations for AI training" \
    --data-categories "personal,sensitive" \
    --metadata '{"source": "customer_support_tickets", "preprocessing": "anonymized"}'
```

### Output Validation

```bash
# Validate agent output for compliance
multimind governance validate-output \
    --output-id "response_123" \
    --content "The customer's account balance is $1,000" \
    --user-id "user_123" \
    --purpose "customer_support"
```

### Anomaly Monitoring

```bash
# Monitor for compliance anomalies
multimind governance monitor-anomalies \
    --start-time "2024-03-01T00:00:00" \
    --severity "high,critical"
```

### Compliance Reports

```bash
# Generate monthly compliance report
multimind governance export-logs \
    --report-id "monthly_2024_03" \
    --period "30d" \
    --format "pdf"
```

### DSAR Handling

```bash
# Export user data for DSAR
multimind governance dsar export \
    --user-id "user_123" \
    --request-id "dsar_2024_001" \
    --format "json"

# Erase user data
multimind governance dsar erase \
    --user-id "user_123" \
    --request-id "erasure_2024_001" \
    --verify
```

### Model Version Approval

```bash
# Request approval for a new model version
multimind governance model-approve \
    --model-id "invoice-processor-v2" \
    --approver "alice@acme.com" \
    --metadata '{"version": "2.0.0", "changes": "Improved accuracy"}'
```

### Third-Party Plugin Vetting

```bash
# Register and vet a new plugin
multimind governance plugin-register \
    --name "sentiment-analyzer" \
    --source "github.com/org/repo" \
    --checks "dependency_scan,license_check,cve_lookup"
```

### Continuous Compliance Testing

```bash
# Run nightly compliance test suite
multimind governance test-run \
    --suite "nightly-safety" \
    --ticket-system "jira" \
    --project "COMPLIANCE"
```

### Embedding Drift Detection

```bash
# Check for embedding drift
multimind governance drift-check \
    --store "prod-embeddings" \
    --threshold 0.15
```

### Risk Score Override

```bash
# Override risk score for a request
multimind governance risk-override \
    --request-id "abc123" \
    --new-score 0.3 \
    --reason "Low sensitivity" \
    --officer-id "compliance_officer_001"
```

### Log Chain Verification

```bash
# Verify tamper-evident log chain
multimind governance audit-verify \
    --chain-id "chain-789" \
    --start-time "2024-03-01T00:00:00" \
    --end-time "2024-03-31T23:59:59"
```

### Policy Management

```bash
# Publish new policy version
multimind governance policy-publish \
    --policy-file "new-gdpr.rego" \
    --version "1.2.0" \
    --metadata '{"author": "compliance_team", "changes": "Updated data retention rules"}'
```

### Incident Response

```bash
# Create and handle incident
multimind governance incident-create \
    --type "policy-violation" \
    --details-file "violation123.json" \
    --severity "high" \
    --playbook "policy_violation_response"
```

### Consent Management

```bash
# Check for expiring consents
multimind governance consent-check \
    --days 7 \
    --channels "email,in_app"
```

### DPIA Assignment

```bash
# Assign DPIA review task
multimind governance dpia-assign \
    --dataset-id "medical-records" \
    --assignee "compliance-team" \
    --priority "high" \
    --due-days 14
```

## Common Use Cases

1. **Dataset Onboarding**
   - Enforce consent checks
   - Tag sensitive data
   - Trigger DPIA if needed

2. **Pre-training Pipeline**
   - Validate data compliance
   - Check for required documentation
   - Ensure proper consent

3. **Agent Response Validation**
   - Check output against schemas
   - Verify policy compliance
   - Log validation results

4. **Live Monitoring**
   - Stream audit logs
   - Detect anomalies
   - Alert on violations

5. **Compliance Reporting**
   - Generate monthly reports
   - Track policy violations
   - Monitor DPIA status

6. **DSAR Processing**
   - Export user data
   - Handle erasure requests
   - Maintain audit trail

## Additional Use Cases

1. **Model Governance**
   - Version approval workflows
   - Change tracking
   - Audit trail maintenance

2. **Plugin Security**
   - Dependency scanning
   - License compliance
   - Vulnerability assessment

3. **Continuous Testing**
   - Automated test suites
   - Integration with ticketing
   - Failure tracking

4. **Drift Monitoring**
   - Embedding distribution analysis
   - Threshold-based alerts
   - Retraining triggers

5. **Risk Management**
   - Score overrides
   - Officer approvals
   - Audit trail

6. **Log Security**
   - Chain verification
   - Tamper detection
   - Historical analysis

7. **Policy Control**
   - Version management
   - Hot reloading
   - Change tracking

8. **Incident Handling**
   - Automated playbooks
   - Severity management
   - Response coordination

9. **Consent Management**
   - Expiry tracking
   - Multi-channel notifications
   - Renewal workflows

10. **DPIA Management**
    - Task assignment
    - Priority handling
    - Due date tracking

## Requirements

- Python 3.8+
- Click
- MultiMind SDK
- Required dependencies (see requirements.txt)

## Notes

- All commands are asynchronous and use the `asyncio` runtime
- Commands can be integrated into CI/CD pipelines
- Use `--help` with any command to see detailed options
- Consider using environment variables for sensitive configuration 