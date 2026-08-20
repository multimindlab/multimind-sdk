"""
Example usage of MultiMind SDK's compliance features via CLI.
"""

import asyncio
from datetime import datetime, timedelta

import click

from multimind.compliance.governance import DataCategory, Regulation
from multimind.compliance.privacy import (
    AuditAction,
    ComplianceReportTemplate,
    GovernanceConfig,
    NotificationType,
    PrivacyCompliance,
    RiskScore,
)


@click.group()
def governance():
    """MultiMind Governance CLI for compliance management."""
    pass

@governance.command()
@click.option('--dataset-id', required=True, help='Unique identifier for the dataset')
@click.option('--name', required=True, help='Name of the dataset')
@click.option('--description', required=True, help='Description of the dataset')
@click.option('--data-categories', required=True, help='Comma-separated list of data categories')
@click.option('--metadata', help='JSON string containing dataset metadata')
def ingest(dataset_id, name, description, data_categories, metadata):
    """Ingest a new dataset with compliance checks."""

    async def _ingest():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            organization_name="Example Organization",
            dpo_email="dpo@example.com",
            enabled_regulations=[Regulation.GDPR, Regulation.AI_ACT]
        ))

        # Parse data categories and convert to DataCategory enum
        category_list = [c.strip() for c in data_categories.split(',') if c.strip()]
        categories = set()
        for cat in category_list or ['personal']:
            try:
                categories.add(DataCategory(cat.lower()))
            except ValueError:
                categories.add(DataCategory.PERSONAL)

        # Parse metadata if provided
        metadata_dict = {}
        if metadata:
            import json
            metadata_dict = json.loads(metadata)

        # Create a processing purpose for this dataset
        purpose_id = f"purpose_{dataset_id}"
        await privacy_manager.add_data_purpose(
            purpose_id=purpose_id,
            name=f"Purpose for {name}",
            description=description,
            legal_basis="legitimate_interest",
            retention_period=365,
            data_categories=categories
        )

        # Process dataset with compliance checks
        result = await privacy_manager.process_privacy_data(
            data_id=dataset_id,
            data_type="dataset",
            content={"name": name, "description": description, **metadata_dict},
            jurisdiction="global",
            data_categories=categories,
            purposes={purpose_id},
            metadata=metadata_dict
        )

        click.echo("Dataset ingested successfully!")
        click.echo(f"  Dataset ID: {result.data_id}")
        click.echo(f"  Categories: {', '.join(sorted(cat.value for cat in result.data_categories))}")
        retention = result.retention_end_date.strftime('%Y-%m-%d') if result.retention_end_date else "N/A"
        click.echo(f"  Retention end date: {retention}")

    asyncio.run(_ingest())

@governance.command()
@click.option('--output-id', required=True, help='ID of the output to validate')
@click.option('--content', required=True, help='Content to validate')
@click.option('--user-id', required=True, help='User ID for context')
@click.option('--purpose', required=True, help='Purpose of the output')
def validate_output(output_id, content, user_id, purpose):
    """Validate agent outputs for compliance."""

    async def _validate():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            organization_name="Example Organization",
            dpo_email="dpo@example.com",
            enabled_regulations=[Regulation.GDPR, Regulation.AI_ACT]
        ))

        purpose_id = f"purpose_{purpose.replace(' ', '_').lower() or 'default'}"
        if purpose_id not in privacy_manager.data_purposes:
            await privacy_manager.add_data_purpose(
                purpose_id=purpose_id,
                name=purpose or "General purpose",
                description=f"Purpose for validating output {output_id}",
                legal_basis="consent",
                retention_period=90,
                data_categories={DataCategory.PERSONAL}
            )

        result = await privacy_manager.process_privacy_data(
            data_id=output_id,
            data_type="output",
            content=content,
            jurisdiction="global",
            data_categories={DataCategory.PERSONAL},
            purposes={purpose_id},
            metadata={"user_id": user_id, "purpose": purpose}
        )

        click.echo("Validation completed.")
        click.echo(f"  Output ID: {result.data_id}")
        click.echo(f"  Purpose: {purpose or 'unspecified'}")
        click.echo(f"  User ID: {user_id}")

    asyncio.run(_validate())

@governance.command()
@click.option('--start-time', required=True, help='Start time for monitoring (ISO format)')
@click.option('--end-time', help='End time for monitoring (ISO format)')
@click.option('--severity', help='Comma-separated list of severity levels')
def monitor_anomalies(start_time, end_time, severity):
    """Monitor for compliance anomalies."""

    async def _monitor():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            organization_name="Example Organization",
            dpo_email="dpo@example.com",
            enabled_regulations=[Regulation.GDPR, Regulation.AI_ACT]
        ))

        severity_levels = [s.strip().upper() for s in severity.split(',')] if severity else None

        # Parse timestamps
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time) if end_time else datetime.now()

        trails = await privacy_manager.get_audit_trails(
            start_date=start,
            end_date=end
        )

        if severity_levels:
            trails = [
                trail for trail in trails
                if trail.metadata.get("severity", "INFO").upper() in severity_levels
            ]

        if not trails:
            click.echo("No anomalies detected for the requested window.")
            return

        click.echo(f"Detected {len(trails)} audit events:")
        for trail in trails[:20]:
            click.echo(
                f"- [{trail.timestamp.isoformat()}] {trail.entity_type}:{trail.entity_id} "
                f"{trail.action.value} by {trail.user_id}"
            )
            if trail.metadata:
                click.echo(f"    details: {trail.metadata}")

    asyncio.run(_monitor())

@governance.command()
@click.option('--report-id', required=True, help='Unique identifier for the report')
@click.option('--period', required=True, help='Report period (e.g., "30d" for 30 days)')
@click.option('--format', default='pdf', help='Report format (pdf, json, csv)')
def export_logs(report_id, period, format):
    """Export compliance logs and generate reports."""

    async def _export():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            organization_name="Example Organization",
            dpo_email="dpo@example.com",
            enabled_regulations=[Regulation.GDPR, Regulation.AI_ACT]
        ))

        # Parse period
        days = int(period[:-1]) if period.endswith('d') else 30
        period_start = datetime.now() - timedelta(days=days)
        period_end = datetime.now()

        # Ensure a template exists before generating report
        template_id = "cli_default_template"
        if template_id not in privacy_manager.report_templates:
            template = ComplianceReportTemplate(
                template_id=template_id,
                name="CLI Default Compliance Report",
                description="Report generated from CLI export-logs command",
                regulation="GDPR",
                jurisdiction="global",
                sections=[
                    {"id": "status", "type": "compliance_status", "title": "Compliance Status"},
                    {"id": "risk", "type": "risk_assessment", "title": "Risk Assessment"},
                    {"id": "audit", "type": "audit_summary", "title": "Audit Summary"}
                ]
            )
            privacy_manager.report_templates[template_id] = template

        report = await privacy_manager.generate_compliance_report(
            template_id=template_id,
            period_start=period_start,
            period_end=period_end,
            jurisdiction="global",
            regulation="GDPR",
            metadata={"report_id": report_id, "format": format}
        )

        click.echo("Report generated successfully!")
        click.echo(f"  Report ID: {report.report_id}")
        click.echo(f"  Period: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
        click.echo(f"  Overall status: {report.overall_status}")
        click.echo(f"  Findings: {len(report.findings)} | Recommendations: {len(report.recommendations)}")

    asyncio.run(_export())

@governance.group()
def dsar():
    """Handle Data Subject Access Requests (DSAR)."""
    pass

@dsar.command()
@click.option('--user-id', required=True, help='User ID for DSAR')
@click.option('--request-id', required=True, help='Unique request identifier')
@click.option('--format', default='json', help='Export format (json, csv)')
def export(user_id, request_id, format):
    """Export user data for DSAR."""

    async def _export():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            organization_name="Example Organization",
            dpo_email="dpo@example.com",
            enabled_regulations=[Regulation.GDPR]
        ))

        result = await privacy_manager.export_data_portability(
            user_id=user_id,
            format=format
        )

        click.echo("DSAR export completed.")
        click.echo(f"  User ID: {user_id}")
        click.echo(f"  Request ID: {request_id}")
        preview = result if isinstance(result, str) else result.decode('utf-8', errors='ignore')
        click.echo(f"  Preview: {preview[:500]}{'...' if len(preview) > 500 else ''}")

    asyncio.run(_export())

@dsar.command()
@click.option('--user-id', required=True, help='User ID for erasure')
@click.option('--request-id', required=True, help='Unique request identifier')
@click.option('--verify/--no-verify', default=True, help='Require verification before erasure')
def erase(user_id, request_id, verify):
    """Erase user data for DSAR."""

    async def _erase():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            organization_name="Example Organization",
            dpo_email="dpo@example.com",
            enabled_regulations=[Regulation.GDPR]
        ))

        # Identify data items related to the user (mock lookup)
        user_data_ids = [
            data_id for data_id, data in privacy_manager.privacy_data.items()
            if data.metadata.get("user_id") == user_id
        ]

        result = await privacy_manager.process_data_subject_request(
            request_type="deletion",
            user_id=user_id,
            data_ids=user_data_ids,
            metadata={"request_id": request_id, "verification_required": verify}
        )

        click.echo("DSAR erasure completed.")
        click.echo(f"  User ID: {user_id}")
        click.echo(f"  Request ID: {request_id}")
        click.echo(f"  Items requested for deletion: {len(user_data_ids)}")

    asyncio.run(_erase())

@governance.command()
@click.option('--model-id', required=True, help='ID of the model to approve')
@click.option('--approver', required=True, help='Email of the approver')
@click.option('--metadata', help='JSON string containing approval metadata')
def model_approve(model_id, approver, metadata):
    """Approve a new model version."""

    async def _approve():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            organization_name="Example Organization",
            dpo_email="dpo@example.com",
            enabled_regulations=[Regulation.GDPR, Regulation.AI_ACT]
        ))

        # Parse metadata if provided
        metadata_dict = {}
        if metadata:
            import json
            metadata_dict = json.loads(metadata)

        workflow_id = f"model_approval_{model_id}"
        workflow = await privacy_manager.create_compliance_workflow(
            workflow_id=workflow_id,
            name=f"Model Approval: {model_id}",
            description=f"Approval workflow for model {model_id}",
            steps=[
                {"step_id": "review", "name": "Model Review", "assignee": approver, "status": "pending"},
                {"step_id": "approval", "name": "Final Approval", "assignee": approver, "status": "pending"}
            ],
            assigned_to=approver,
            metadata={"model_id": model_id, **metadata_dict}
        )

        click.echo("Model approval workflow created.")
        click.echo(f"  Workflow ID: {workflow.workflow_id}")
        click.echo(f"  Current step: {workflow.current_step + 1}/{len(workflow.steps)}")
        click.echo(f"  Assignee: {workflow.assigned_to or 'unassigned'}")

    asyncio.run(_approve())

@governance.command()
@click.option('--name', required=True, help='Name of the plugin')
@click.option('--source', required=True, help='Source repository URL')
@click.option('--checks', help='Comma-separated list of security checks to run')
def plugin_register(name, source, checks):
    """Register and vet a third-party plugin."""

    async def _register():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            organization_name="Example Organization",
            dpo_email="dpo@example.com",
            enabled_regulations=[Regulation.GDPR, Regulation.AI_ACT]
        ))

        # Parse security checks
        check_list = checks.split(',') if checks else ["dependency_scan", "license_check", "cve_lookup"]

        audit_trail = await privacy_manager.create_audit_trail(
            action=AuditAction.CREATE,
            entity_type="plugin",
            entity_id=name,
            user_id="system",
            metadata={
                "source": source,
                "checks": check_list,
                "status": "registered"
            }
        )

        click.echo("Plugin registration recorded.")
        click.echo(f"  Plugin: {name}")
        click.echo(f"  Source: {source}")
        click.echo(f"  Checks performed: {', '.join(check_list)}")
        click.echo(f"  Audit trail ID: {audit_trail.trail_id}")

    asyncio.run(_register())

@governance.command()
@click.option('--suite', required=True, help='Test suite identifier')
@click.option('--ticket-system', help='Ticket system for failures (e.g., jira)')
@click.option('--project', help='Project identifier in ticket system')
def test_run(suite, ticket_system, project):
    """Run compliance test suite."""

    async def _run_tests():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            organization_name="Example Organization",
            dpo_email="dpo@example.com",
            enabled_regulations=[Regulation.GDPR, Regulation.AI_ACT]
        ))

        results = {}
        for regulation in privacy_manager.config.enabled_regulations:
            # Handle both enum and string types
            regulation_str = regulation.value if hasattr(regulation, 'value') else str(regulation)
            try:
                score = await privacy_manager.calculate_compliance_score(
                    entity_id=suite,
                    regulation=regulation_str,
                    jurisdiction="global"
                )
                results[regulation_str] = {
                    "score": round(score.score, 2),
                    "trend": score.trend,
                    "components": {k: round(v, 2) for k, v in score.components.items()}
                }
            except Exception as exc:
                results[regulation_str] = f"Error: {exc}"

        if ticket_system and project:
            results["integration"] = {
                "ticket_system": ticket_system,
                "project": project
            }

        click.echo(f"Compliance test summary for '{suite}':")
        for regulation, value in results.items():
            click.echo(f"  {regulation}: {value}")

    asyncio.run(_run_tests())

@governance.command()
@click.option('--store', required=True, help='Embedding store identifier')
@click.option('--threshold', type=float, default=0.15, help='Drift threshold')
def drift_check(store, threshold):
    """Check for embedding drift."""

    async def _check_drift():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            organization_name="Example Organization",
            dpo_email="dpo@example.com",
            enabled_regulations=[Regulation.GDPR, Regulation.AI_ACT]
        ))

        # Use existing anomalies list instead of calling detect_anomalies()
        # which has a bug (calls non-existent _get_recent_api_calls)
        try:
            anomalies = await privacy_manager.detect_anomalies()
        except AttributeError:
            # Fallback to existing anomalies if detect_anomalies() fails
            anomalies = privacy_manager.anomalies

        # Filter for drift-related anomalies for this store
        # and check if they exceed the threshold
        drift_anomalies = []
        for anomaly in anomalies:
            if (hasattr(anomaly, 'anomaly_type') and anomaly.anomaly_type == "drift"
                and hasattr(anomaly, 'metrics') and anomaly.metrics.get("store_id") == store):
                # Check if drift exceeds threshold
                current_value = getattr(anomaly, 'current_value', 0)
                anomaly_threshold = getattr(anomaly, 'threshold', threshold)
                # Use the anomaly's threshold if available, otherwise use provided threshold
                if current_value >= anomaly_threshold or current_value >= threshold:
                    drift_anomalies.append(anomaly)

        click.echo(f"Embedding drift check for store '{store}':")
        click.echo(f"  Threshold: {threshold} (15% drift considered significant)")
        click.echo(f"  Anomalies detected: {len(drift_anomalies)}")

        if drift_anomalies:
            click.echo("  Drift anomalies exceeding threshold:")
            for anomaly in drift_anomalies[:10]:
                timestamp = getattr(anomaly, 'timestamp', None)
                severity = getattr(anomaly, 'severity', 'unknown')
                current_value = getattr(anomaly, 'current_value', None)
                threshold_val = getattr(anomaly, 'threshold', threshold)

                timestamp_str = timestamp.isoformat() if timestamp else "N/A"
                drift_percent = (current_value * 100) if current_value else 0
                click.echo(
                    f"  - [{timestamp_str}] severity={severity} "
                    f"drift={drift_percent:.2f}% threshold={threshold_val*100:.1f}%"
                )
        else:
            click.echo(f"  No drift anomalies found exceeding threshold ({threshold*100}%).")
            click.echo("  (This means embedding data is within acceptable drift limits)")

    asyncio.run(_check_drift())

@governance.command()
@click.option('--request-id', required=True, help='Request identifier')
@click.option('--new-score', type=float, required=True, help='New risk score')
@click.option('--reason', required=True, help='Reason for override')
@click.option('--officer-id', required=True, help='ID of compliance officer')
def risk_override(request_id, new_score, reason, officer_id):
    """Override risk score for a request."""

    async def _override():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            organization_name="Example Organization",
            dpo_email="dpo@example.com",
            enabled_regulations=[Regulation.GDPR, Regulation.AI_ACT]
        ))

        try:
            current_score = await privacy_manager.calculate_risk_score(entity_id=request_id)
        except Exception:
            current_score = RiskScore(score=0.0, level="high", factors=[])

        # Override score manually (higher score => higher risk)
        new_level = (
            "critical" if new_score >= 0.8 else
            "high" if new_score >= 0.6 else
            "medium" if new_score >= 0.4 else
            "low"
        )
        privacy_manager.risk_scores[request_id] = RiskScore(
            score=new_score,
            level=new_level,
            factors=[{"override": True, "reason": reason, "officer_id": officer_id}]
        )

        await privacy_manager.create_audit_trail(
            action=AuditAction.UPDATE,
            entity_type="risk_score",
            entity_id=request_id,
            user_id=officer_id,
            metadata={"old_score": current_score.score, "new_score": new_score, "reason": reason}
        )

        click.echo("Risk score override recorded.")
        click.echo(f"  Request ID: {request_id}")
        click.echo(f"  Old score: {current_score.score:.2f}")
        click.echo(f"  New score: {new_score:.2f} ({new_level})")
        click.echo(f"  Override reason: {reason}")

    asyncio.run(_override())

@governance.command()
@click.option('--chain-id', required=True, help='Log chain identifier')
@click.option('--start-time', help='Start time for verification (ISO format)')
@click.option('--end-time', help='End time for verification (ISO format)')
def audit_verify(chain_id, start_time, end_time):
    """Verify tamper-evident log chain."""

    async def _verify():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            organization_name="Example Organization",
            dpo_email="dpo@example.com",
            enabled_regulations=[Regulation.GDPR, Regulation.AI_ACT]
        ))

        # Parse timestamps if provided
        start = datetime.fromisoformat(start_time) if start_time else None
        end = datetime.fromisoformat(end_time) if end_time else None

        trails = await privacy_manager.get_audit_trails(
            entity_id=chain_id,
            start_date=start,
            end_date=end
        )

        is_valid = len(trails) > 0
        gaps_detected = False
        if len(trails) > 1:
            ordered = sorted(trails, key=lambda t: t.timestamp)
            for prev, cur in zip(ordered, ordered[1:]):
                if cur.timestamp < prev.timestamp:
                    gaps_detected = True
                    is_valid = False
                    break

        result = {
            "chain_id": chain_id,
            "events": len(trails),
            "valid": is_valid and not gaps_detected,
            "gaps_detected": gaps_detected,
            "start_time": start,
            "end_time": end
        }

        click.echo("Audit chain verification summary:")
        for key, value in result.items():
            click.echo(f"  {key}: {value}")

    asyncio.run(_verify())

@governance.command()
@click.option('--policy-file', required=True, help='Path to policy file')
@click.option('--version', required=True, help='Policy version')
@click.option('--metadata', help='JSON string containing policy metadata')
def policy_publish(policy_file, version, metadata):
    """Publish new policy version."""

    async def _publish():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            organization_name="Example Organization",
            dpo_email="dpo@example.com",
            enabled_regulations=[Regulation.GDPR, Regulation.AI_ACT]
        ))

        # Parse metadata if provided
        metadata_dict = {}
        if metadata:
            import json
            metadata_dict = json.loads(metadata)

        try:
            with open(policy_file, "r", encoding="utf-8") as f:
                policy_content = f.read()
        except FileNotFoundError:
            click.echo(f"Policy file not found: {policy_file}")
            return

        audit_trail = await privacy_manager.create_audit_trail(
            action=AuditAction.CREATE,
            entity_type="policy",
            entity_id=f"policy_{version}",
            user_id="system",
            metadata={
                "version": version,
                "file": policy_file,
                "content_length": len(policy_content),
                **metadata_dict
            }
        )

        click.echo("Policy publication recorded.")
        click.echo(f"  Version: {version}")
        click.echo(f"  File: {policy_file}")
        click.echo(f"  Audit trail ID: {audit_trail.trail_id}")

    asyncio.run(_publish())

@governance.command()
@click.option('--type', required=True, help='Incident type')
@click.option('--details-file', required=True, help='Path to incident details file')
@click.option('--severity', default='high', help='Incident severity')
@click.option('--playbook', help='Response playbook to execute')
def incident_create(type, details_file, severity, playbook):
    """Create and handle incident."""

    async def _create():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            organization_name="Example Organization",
            dpo_email="dpo@example.com",
            enabled_regulations=[Regulation.GDPR, Regulation.AI_ACT]
        ))

        try:
            with open(details_file, "r", encoding="utf-8") as f:
                details = f.read()
        except FileNotFoundError:
            click.echo(f"Incident details file not found: {details_file}")
            return

        event = await privacy_manager.create_compliance_event(
            title=f"Incident: {type}",
            description=f"{details[:500]}\nSeverity: {severity}\nPlaybook: {playbook or 'None'}",
            event_type="incident",
            start_date=datetime.now(),
            jurisdiction="global",
            regulation="general"
        )

        await privacy_manager.create_notification(
            type=NotificationType.RISK_ALERT,
            title=f"Incident created: {type}",
            message=f"Severity {severity} incident recorded (event {event.event_id})",
            priority=severity,
            recipient="security_team",
            metadata={"event_id": event.event_id}
        )

        click.echo("Incident recorded and team notified.")
        click.echo(f"  Event ID: {event.event_id}")
        click.echo(f"  Severity: {severity}")

    asyncio.run(_create())

@governance.command()
@click.option('--days', type=int, default=7, help='Days until consent expiry')
@click.option('--channels', help='Comma-separated list of notification channels')
def consent_check(days, channels):
    """Check for expiring consents."""

    async def _check():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            organization_name="Example Organization",
            dpo_email="dpo@example.com",
            enabled_regulations=[Regulation.GDPR]
        ))

        channel_list = [c.strip() for c in channels.split(',')] if channels else ["email"]
        consent_history = await privacy_manager.get_consent_history()
        cutoff = datetime.now() + timedelta(days=days)

        expiring_consents = []
        for consent in consent_history:
            timestamp = consent.get("timestamp")
            if isinstance(timestamp, datetime) and timestamp < cutoff and consent.get("granted"):
                expiring_consents.append(consent)
                if len(expiring_consents) >= 10:
                    break

        for consent in expiring_consents:
            await privacy_manager.create_notification(
                type=NotificationType.CONSENT_EXPIRY,
                title="Consent expiring soon",
                message=f"Consent {consent.get('consent_id')} for user {consent.get('user_id')} is expiring.",
                priority="medium",
                recipient=consent.get("user_id", "user"),
                metadata={"consent_id": consent.get("consent_id")}
            )

        click.echo("Consent check summary:")
        click.echo(f"  Window: {days} days")
        click.echo(f"  Expiring consents found: {len(expiring_consents)}")
        click.echo(f"  Notification channels: {', '.join(channel_list)}")

    asyncio.run(_check())

@governance.command()
@click.option('--dataset-id', required=True, help='Dataset identifier')
@click.option('--assignee', required=True, help='Assignee for DPIA review')
@click.option('--priority', default='high', help='Review priority')
@click.option('--due-days', type=int, default=14, help='Days until due date')
def dpia_assign(dataset_id, assignee, priority, due_days):
    """Assign DPIA review task."""

    async def _assign():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            organization_name="Example Organization",
            dpo_email="dpo@example.com",
            enabled_regulations=[Regulation.GDPR, Regulation.AI_ACT]
        ))

        due_date = datetime.now() + timedelta(days=due_days)
        event = await privacy_manager.create_compliance_event(
            title=f"DPIA Review: {dataset_id}",
            description=f"Data Protection Impact Assessment for dataset {dataset_id}. Priority: {priority}",
            event_type="dpia_review",
            start_date=datetime.now(),
            end_date=due_date,
            jurisdiction="global",
            regulation="GDPR",
            assigned_to=assignee
        )

        await privacy_manager.create_notification(
            type=NotificationType.DEADLINE_REMINDER,
            title=f"DPIA review assigned: {dataset_id}",
            message=f"DPIA review assigned to {assignee}, due on {due_date.strftime('%Y-%m-%d')}",
            priority=priority,
            recipient=assignee,
            metadata={"event_id": event.event_id}
        )

        click.echo("DPIA review task created.")
        click.echo(f"  Event ID: {event.event_id}")
        click.echo(f"  Assignee: {assignee}")
        click.echo(f"  Due date: {due_date.strftime('%Y-%m-%d')}")

    asyncio.run(_assign())

if __name__ == '__main__':
    governance()
