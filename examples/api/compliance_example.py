"""
Example usage of MultiMind SDK's compliance features via API.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from multimind.compliance.privacy import (
    PrivacyCompliance,
    GovernanceConfig,
    DataCategory,
    AuditAction
)

async def ingest_dataset_example():
    """Example of onboarding a new dataset with compliance checks."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "AI_ACT"]
    ))
    
    # Ingest dataset with compliance checks
    dataset = await privacy_manager.ingest_dataset(
        dataset_id="customer_support_2024",
        name="Customer Support Dataset",
        description="Customer support conversations for AI training",
        data_categories={DataCategory.PERSONAL, DataCategory.SENSITIVE},
        metadata={
            "source": "customer_support_tickets",
            "preprocessing": "anonymized",
            "version": "1.0"
        }
    )
    
    # Check if DPIA is needed
    dpia_required = await privacy_manager.check_dpia_requirement(
        dataset_id=dataset.id,
        risk_factors={
            "personal_data": True,
            "sensitive_data": True,
            "automated_decision_making": True
        }
    )
    
    return {
        "dataset": dataset,
        "dpia_required": dpia_required
    }

async def validate_agent_output_example():
    """Example of validating agent outputs for compliance."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "AI_ACT"]
    ))
    
    # Validate agent output
    validation = await privacy_manager.validate_output(
        output_id="response_123",
        content="The customer's account balance is $1,000",
        context={
            "user_id": "user_123",
            "purpose": "customer_support",
            "data_categories": {DataCategory.PERSONAL, DataCategory.FINANCIAL}
        }
    )
    
    # Check for policy violations
    violations = await privacy_manager.check_policy_violations(
        output_id=validation.id,
        policies=["data_minimization", "purpose_limitation"]
    )
    
    return {
        "validation": validation,
        "violations": violations
    }

async def monitor_anomalies_example():
    """Example of monitoring for compliance anomalies."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "AI_ACT"]
    ))
    
    # Set up anomaly detection
    detector = await privacy_manager.setup_anomaly_detection(
        detector_id="compliance_monitor_001",
        patterns=[
            {
                "type": "data_access",
                "threshold": 100,
                "window": "1h"
            },
            {
                "type": "policy_violation",
                "severity": "high",
                "window": "24h"
            }
        ]
    )
    
    # Stream audit logs
    async for event in privacy_manager.stream_audit_logs(
        start_time=datetime.now() - timedelta(hours=1),
        filters={
            "action_types": [AuditAction.ACCESS, AuditAction.MODIFY],
            "severity": ["high", "critical"]
        }
    ):
        # Process events in real-time
        await process_audit_event(event)
    
    return detector

async def generate_compliance_report_example():
    """Example of generating compliance reports."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "AI_ACT"]
    ))
    
    # Generate monthly report
    report = await privacy_manager.generate_report(
        report_id="monthly_2024_03",
        period_start=datetime.now() - timedelta(days=30),
        period_end=datetime.now(),
        sections=[
            "risk_assessment",
            "policy_violations",
            "dpia_status",
            "data_processing"
        ],
        format="pdf"
    )
    
    return report

async def handle_dsar_example():
    """Example of handling Data Subject Access Requests (DSAR)."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR"]
    ))
    
    # Export user data for DSAR
    export = await privacy_manager.export_user_data(
        user_id="user_123",
        request_id="dsar_2024_001",
        data_types=[
            "interaction_history",
            "personal_data",
            "consent_records"
        ],
        format="json"
    )
    
    # Handle erasure request
    erasure = await privacy_manager.erase_user_data(
        user_id="user_123",
        request_id="erasure_2024_001",
        data_types=[
            "personal_data",
            "interaction_history"
        ],
        verification_required=True
    )
    
    return {
        "export": export,
        "erasure": erasure
    }

async def model_approval_example():
    """Example of model version approval workflow."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "AI_ACT"]
    ))
    
    # Request model approval
    approval = await privacy_manager.request_model_approval(
        model_id="invoice-processor-v2",
        approver_email="alice@acme.com",
        metadata={
            "version": "2.0.0",
            "changes": "Improved accuracy and reduced bias",
            "risk_assessment": "low"
        }
    )
    
    # Record approval in audit logs
    audit = await privacy_manager.record_approval(
        approval_id=approval.id,
        approver="alice@acme.com",
        timestamp=datetime.now(),
        signature="digital_signature_here"
    )
    
    return {
        "approval": approval,
        "audit": audit
    }

async def plugin_vetting_example():
    """Example of third-party plugin vetting process."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "AI_ACT"]
    ))
    
    # Register and vet plugin
    plugin = await privacy_manager.register_plugin(
        name="sentiment-analyzer",
        source="github.com/org/repo",
        checks={
            "dependency_scan": True,
            "license_check": True,
            "cve_lookup": True
        }
    )
    
    # Run security checks
    security_report = await privacy_manager.run_security_checks(
        plugin_id=plugin.id,
        checks=["dependency_scan", "license_check", "cve_lookup"]
    )
    
    return {
        "plugin": plugin,
        "security_report": security_report
    }

async def compliance_testing_example():
    """Example of continuous compliance testing."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "AI_ACT"]
    ))
    
    # Run compliance test suite
    test_suite = await privacy_manager.run_compliance_tests(
        suite_id="nightly-safety",
        tests=[
            {
                "name": "adversarial_prompts",
                "type": "safety",
                "threshold": 0.95
            },
            {
                "name": "bias_detection",
                "type": "fairness",
                "threshold": 0.90
            }
        ],
        integration={
            "ticket_system": "jira",
            "project": "COMPLIANCE"
        }
    )
    
    return test_suite

async def drift_detection_example():
    """Example of embedding drift detection."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "AI_ACT"]
    ))
    
    # Check for embedding drift
    drift_report = await privacy_manager.check_embedding_drift(
        store_id="prod-embeddings",
        threshold=0.15,
        metrics={
            "cosine_similarity": True,
            "distribution_shift": True
        }
    )
    
    # Handle drift if detected
    if drift_report.drift_detected:
        await privacy_manager.trigger_retraining(
            model_id=drift_report.model_id,
            reason="embedding_drift",
            priority="high"
        )
    
    return drift_report

async def risk_override_example():
    """Example of runtime risk score override."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "AI_ACT"]
    ))
    
    # Override risk score
    override = await privacy_manager.override_risk_score(
        request_id="abc123",
        new_score=0.3,
        reason="Low sensitivity",
        officer_id="compliance_officer_001"
    )
    
    # Record override in audit trail
    audit = await privacy_manager.record_risk_override(
        override_id=override.id,
        original_score=0.8,
        new_score=0.3,
        reason="Low sensitivity"
    )
    
    return {
        "override": override,
        "audit": audit
    }

async def log_verification_example():
    """Example of tamper-evident log verification."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "AI_ACT"]
    ))
    
    # Verify log chain integrity
    verification = await privacy_manager.verify_log_chain(
        chain_id="chain-789",
        verification_type="merkle",
        start_time=datetime.now() - timedelta(days=30),
        end_time=datetime.now()
    )
    
    return verification

async def policy_management_example():
    """Example of policy roll-out and versioning."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "AI_ACT"]
    ))
    
    # Publish new policy
    policy = await privacy_manager.publish_policy(
        policy_file="new-gdpr.rego",
        version="1.2.0",
        metadata={
            "author": "compliance_team",
            "changes": "Updated data retention rules"
        }
    )
    
    # Hot-reload policies
    reload = await privacy_manager.reload_policies(
        policy_id=policy.id,
        services=["inference_service_1", "inference_service_2"]
    )
    
    return {
        "policy": policy,
        "reload": reload
    }

async def incident_response_example():
    """Example of incident response playbook trigger."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "AI_ACT"]
    ))
    
    # Create incident
    incident = await privacy_manager.create_incident(
        type="policy-violation",
        details_file="violation123.json",
        severity="high",
        playbook="policy_violation_response"
    )
    
    # Execute playbook
    playbook_result = await privacy_manager.execute_playbook(
        incident_id=incident.id,
        steps=[
            "isolate_model",
            "notify_stakeholders",
            "collect_evidence"
        ]
    )
    
    return {
        "incident": incident,
        "playbook_result": playbook_result
    }

async def consent_expiry_example():
    """Example of consent expiry notification."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR"]
    ))
    
    # Check for expiring consents
    expiring_consents = await privacy_manager.check_consent_expiry(
        days_until_expiry=7,
        notification_channels=["email", "in_app"]
    )
    
    # Send notifications
    notifications = await privacy_manager.send_consent_notifications(
        consents=expiring_consents,
        template="consent_renewal",
        channels=["email"]
    )
    
    return {
        "expiring_consents": expiring_consents,
        "notifications": notifications
    }

async def dpia_assignment_example():
    """Example of automated DPIA review assignment."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "AI_ACT"]
    ))
    
    # Assign DPIA review
    assignment = await privacy_manager.assign_dpia_review(
        dataset_id="medical-records",
        assignee="compliance-team",
        priority="high",
        due_date=datetime.now() + timedelta(days=14)
    )
    
    # Track review status
    status = await privacy_manager.track_dpia_status(
        assignment_id=assignment.id,
        status="in_review",
        comments="Initial assessment in progress"
    )
    
    return {
        "assignment": assignment,
        "status": status
    }

async def run_all_examples():
    """Run all compliance API examples."""
    
    results = {
        "dataset_ingest": await ingest_dataset_example(),
        "output_validation": await validate_agent_output_example(),
        "anomaly_monitoring": await monitor_anomalies_example(),
        "compliance_report": await generate_compliance_report_example(),
        "dsar_handling": await handle_dsar_example(),
        "model_approval": await model_approval_example(),
        "plugin_vetting": await plugin_vetting_example(),
        "compliance_testing": await compliance_testing_example(),
        "drift_detection": await drift_detection_example(),
        "risk_override": await risk_override_example(),
        "log_verification": await log_verification_example(),
        "policy_management": await policy_management_example(),
        "incident_response": await incident_response_example(),
        "consent_expiry": await consent_expiry_example(),
        "dpia_assignment": await dpia_assignment_example()
    }
    
    return results

if __name__ == "__main__":
    asyncio.run(run_all_examples()) 