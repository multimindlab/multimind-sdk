"""
Example usage scenarios for the privacy compliance module.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Set
from multimind.compliance import (
    PrivacyCompliance,
    GovernanceConfig,
    DataCategory,
    NotificationType,
    AuditAction,
    ComplianceStatus
)

async def example_data_purpose_management():
    """Example of managing data purposes and processing privacy data."""
    
    # Initialize privacy compliance manager
    config = GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "CCPA", "PDPA"]
    )
    privacy_manager = PrivacyCompliance(config=config)
    
    # Create data purposes
    marketing_purpose = await privacy_manager.add_data_purpose(
        purpose_id="marketing_001",
        name="Marketing Communications",
        description="Process user data for marketing communications",
        legal_basis="consent",
        retention_period=365,  # 1 year
        data_categories={DataCategory.PERSONAL, DataCategory.CONTACT}
    )
    
    analytics_purpose = await privacy_manager.add_data_purpose(
        purpose_id="analytics_001",
        name="Usage Analytics",
        description="Analyze user behavior and improve services",
        legal_basis="legitimate_interest",
        retention_period=730,  # 2 years
        data_categories={DataCategory.USAGE, DataCategory.TECHNICAL}
    )
    
    # Process privacy-sensitive data
    user_data = await privacy_manager.process_privacy_data(
        data_id="user_123",
        data_type="user_profile",
        content={
            "name": "John Doe",
            "email": "john@example.com",
            "preferences": {"marketing": True}
        },
        jurisdiction="EU",
        data_categories={DataCategory.PERSONAL, DataCategory.CONTACT},
        purposes={"marketing_001", "analytics_001"},
        consent_status={"marketing_001": True, "analytics_001": True}
    )
    
    return user_data

async def example_compliance_monitoring():
    """Example of compliance monitoring and risk assessment."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "CCPA", "PDPA"]
    ))
    
    # Calculate risk score
    risk_score = await privacy_manager.calculate_risk_score(
        entity_id="system_001",
        entity_type="system"
    )
    
    # Create compliance dashboard
    dashboard = await privacy_manager.create_compliance_dashboard(
        dashboard_id="main_dashboard",
        name="Main Compliance Dashboard",
        description="Overview of compliance status and risks",
        refresh_interval=3600  # 1 hour
    )
    
    # Update dashboard metrics
    metrics = await privacy_manager.update_dashboard_metrics("main_dashboard")
    
    # Monitor risk thresholds
    notifications = await privacy_manager.monitor_risk_thresholds()
    
    return {
        "risk_score": risk_score,
        "dashboard_metrics": metrics,
        "notifications": notifications
    }

async def example_audit_and_reporting():
    """Example of audit trail and compliance reporting."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "CCPA", "PDPA"]
    ))
    
    # Create audit trail
    audit_trail = await privacy_manager.create_audit_trail(
        action=AuditAction.ACCESS,
        entity_type="user_data",
        entity_id="user_123",
        user_id="admin_001",
        changes={"accessed_fields": ["email", "preferences"]},
        ip_address="192.168.1.1",
        user_agent="Mozilla/5.0"
    )
    
    # Create report template
    template = await privacy_manager.create_report_template(
        template_id="quarterly_report",
        name="Quarterly Compliance Report",
        description="Quarterly compliance status and findings",
        regulation="GDPR",
        jurisdiction="EU",
        sections=[
            {
                "id": "compliance_status",
                "type": "compliance_status",
                "title": "Compliance Status"
            },
            {
                "id": "risk_assessment",
                "type": "risk_assessment",
                "title": "Risk Assessment"
            },
            {
                "id": "audit_summary",
                "type": "audit_summary",
                "title": "Audit Summary"
            }
        ]
    )
    
    # Generate compliance report
    report = await privacy_manager.generate_compliance_report(
        template_id="quarterly_report",
        period_start=datetime.now() - timedelta(days=90),
        period_end=datetime.now(),
        jurisdiction="EU",
        regulation="GDPR"
    )
    
    return {
        "audit_trail": audit_trail,
        "report": report
    }

async def example_anomaly_detection():
    """Example of anomaly detection and policy violation alerts."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "CCPA", "PDPA"]
    ))
    
    # Detect anomalies
    anomalies = await privacy_manager.detect_anomalies()
    
    # Create policy violation alert
    alert = await privacy_manager.create_policy_alert(
        rule_id="data_retention_001",
        severity="high",
        description="Data retention period exceeded",
        context={
            "data_id": "user_123",
            "retention_period": 365,
            "current_age": 400
        },
        notification_channels=["email", "slack"]
    )
    
    return {
        "anomalies": anomalies,
        "alert": alert
    }

async def example_compliance_workflow():
    """Example of compliance workflow and remediation actions."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "CCPA", "PDPA"]
    ))
    
    # Create remediation workflow
    workflow = await privacy_manager.create_remediation_workflow(
        workflow_id="data_cleanup_001",
        name="Data Cleanup Workflow",
        description="Automated workflow for cleaning up expired data",
        trigger_type="retention",
        trigger_conditions={
            "days_threshold": 30
        },
        steps=[
            {
                "type": "data_deletion",
                "parameters": {
                    "data_ids": ["user_123", "user_456"]
                }
            },
            {
                "type": "consent_obtainment",
                "parameters": {
                    "data_ids": ["user_789"]
                }
            }
        ],
        priority="high"
    )
    
    # Check workflow triggers
    triggered_workflows = await privacy_manager.check_workflow_triggers()
    
    # Create remediation action
    action = await privacy_manager.create_remediation_action(
        action_type="data_deletion",
        target_data=["user_123"],
        priority="high",
        parameters={"reason": "retention_period_exceeded"}
    )
    
    # Execute remediation action
    result = await privacy_manager.execute_remediation_action(action.action_id)
    
    return {
        "workflow": workflow,
        "triggered_workflows": triggered_workflows,
        "remediation_result": result
    }

async def example_compliance_training():
    """Example of compliance training and tracking."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "CCPA", "PDPA"]
    ))
    
    # Create compliance training
    training = await privacy_manager.create_compliance_training(
        training_id="privacy_101",
        title="Privacy Compliance Basics",
        description="Introduction to privacy regulations and compliance",
        modules=[
            {
                "id": "module_1",
                "title": "GDPR Overview",
                "duration": 30
            },
            {
                "id": "module_2",
                "title": "Data Protection Principles",
                "duration": 45
            }
        ],
        target_audience=["employees", "contractors"],
        duration=120,  # 2 hours
        completion_criteria={
            "required_modules": ["module_1", "module_2"],
            "minimum_percentage": 80
        }
    )
    
    # Track training completion
    completion = await privacy_manager.track_training_completion(
        training_id="privacy_101",
        user_id="employee_001",
        completed_modules=["module_1", "module_2"],
        completion_date=datetime.now()
    )
    
    return {
        "training": training,
        "completion": completion
    }

async def example_ai_model_governance():
    """Example of AI model governance and compliance tracking."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "AI_ACT", "ISO27001"]
    ))
    
    # Track model training data
    training_data = await privacy_manager.track_training_data(
        model_id="gpt4_finetune_001",
        dataset_id="customer_support_2024",
        data_categories={DataCategory.PERSONAL, DataCategory.SENSITIVE},
        consent_status=True,
        retention_period=365,
        metadata={
            "source": "customer_support_tickets",
            "preprocessing": "anonymized",
            "version": "1.0"
        }
    )
    
    # Monitor model performance and bias
    bias_metrics = await privacy_manager.monitor_model_bias(
        model_id="gpt4_finetune_001",
        metrics={
            "demographic_parity": 0.85,
            "equal_opportunity": 0.92,
            "disparate_impact": 0.88
        },
        threshold=0.8
    )
    
    # Track model versioning and compliance
    model_version = await privacy_manager.track_model_version(
        model_id="gpt4_finetune_001",
        version="1.1",
        compliance_status={
            "ai_act_risk_level": "high",
            "gdpr_compliant": True,
            "bias_assessment": "passed"
        },
        documentation={
            "technical_doc": "https://docs.example.com/model/001",
            "impact_assessment": "https://docs.example.com/impact/001"
        }
    )
    
    return {
        "training_data": training_data,
        "bias_metrics": bias_metrics,
        "model_version": model_version
    }

async def example_data_protection_impact_assessment():
    """Example of conducting a Data Protection Impact Assessment (DPIA)."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "AI_ACT"]
    ))
    
    # Initiate DPIA
    dpia = await privacy_manager.initiate_dpia(
        project_id="ai_chatbot_001",
        name="Customer Support AI Chatbot",
        description="AI-powered chatbot for customer support",
        risk_level="high",
        data_categories={DataCategory.PERSONAL, DataCategory.SENSITIVE},
        stakeholders=["dpo", "legal", "security"]
    )
    
    # Assess risks
    risk_assessment = await privacy_manager.assess_dpia_risks(
        dpia_id=dpia.id,
        risks=[
            {
                "type": "data_breach",
                "likelihood": "medium",
                "impact": "high",
                "mitigation": "encryption_at_rest"
            },
            {
                "type": "bias",
                "likelihood": "low",
                "impact": "medium",
                "mitigation": "regular_bias_testing"
            }
        ]
    )
    
    # Document controls
    controls = await privacy_manager.document_dpia_controls(
        dpia_id=dpia.id,
        controls=[
            {
                "type": "technical",
                "name": "Data Encryption",
                "description": "End-to-end encryption for all data",
                "status": "implemented"
            },
            {
                "type": "organizational",
                "name": "Regular Audits",
                "description": "Quarterly security audits",
                "status": "planned"
            }
        ]
    )
    
    return {
        "dpia": dpia,
        "risk_assessment": risk_assessment,
        "controls": controls
    }

async def example_consent_management():
    """Example of managing user consent and preferences."""
    
    privacy_manager = PrivacyCompliance(config=GovernanceConfig(
        organization_id="org_123",
        jurisdiction="global",
        regulations=["GDPR", "CCPA"]
    ))
    
    # Record user consent
    consent = await privacy_manager.record_consent(
        user_id="user_123",
        purposes={
            "marketing": {
                "granted": True,
                "timestamp": datetime.now(),
                "version": "1.0",
                "channel": "web"
            },
            "analytics": {
                "granted": True,
                "timestamp": datetime.now(),
                "version": "1.0",
                "channel": "web"
            }
        },
        preferences={
            "email_frequency": "weekly",
            "data_sharing": "limited"
        }
    )
    
    # Verify consent for data processing
    verification = await privacy_manager.verify_consent(
        user_id="user_123",
        purpose="marketing",
        data_categories={DataCategory.PERSONAL, DataCategory.CONTACT}
    )
    
    # Handle consent withdrawal
    withdrawal = await privacy_manager.handle_consent_withdrawal(
        user_id="user_123",
        purpose="marketing",
        timestamp=datetime.now(),
        reason="user_request"
    )
    
    return {
        "consent": consent,
        "verification": verification,
        "withdrawal": withdrawal
    }

async def run_all_examples():
    """Run all example scenarios."""
    
    results = {
        "data_purpose": await example_data_purpose_management(),
        "compliance_monitoring": await example_compliance_monitoring(),
        "audit_reporting": await example_audit_and_reporting(),
        "anomaly_detection": await example_anomaly_detection(),
        "compliance_workflow": await example_compliance_workflow(),
        "compliance_training": await example_compliance_training(),
        "ai_model_governance": await example_ai_model_governance(),
        "data_protection_impact_assessment": await example_data_protection_impact_assessment(),
        "consent_management": await example_consent_management()
    }
    
    return results 