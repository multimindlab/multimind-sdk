"""
Example usage of MultiMind SDK's compliance features via CLI.
"""

import click
import asyncio
from datetime import datetime, timedelta
from multimind.compliance.privacy import PrivacyCompliance, GovernanceConfig

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
            jurisdiction="global",
            regulations=["GDPR", "AI_ACT"]
        ))
        
        # Parse data categories
        categories = set(data_categories.split(','))
        
        # Parse metadata if provided
        metadata_dict = {}
        if metadata:
            import json
            metadata_dict = json.loads(metadata)
        
        # Ingest dataset
        result = await privacy_manager.ingest_dataset(
            dataset_id=dataset_id,
            name=name,
            description=description,
            data_categories=categories,
            metadata=metadata_dict
        )
        
        click.echo(f"Dataset ingested successfully: {result}")
    
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
            jurisdiction="global",
            regulations=["GDPR", "AI_ACT"]
        ))
        
        result = await privacy_manager.validate_output(
            output_id=output_id,
            content=content,
            context={
                "user_id": user_id,
                "purpose": purpose
            }
        )
        
        click.echo(f"Validation result: {result}")
    
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
            jurisdiction="global",
            regulations=["GDPR", "AI_ACT"]
        ))
        
        # Parse severity levels
        severity_levels = severity.split(',') if severity else None
        
        # Parse timestamps
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time) if end_time else datetime.now()
        
        async for event in privacy_manager.stream_audit_logs(
            start_time=start,
            end_time=end,
            filters={"severity": severity_levels} if severity_levels else None
        ):
            click.echo(f"Anomaly detected: {event}")
    
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
            jurisdiction="global",
            regulations=["GDPR", "AI_ACT"]
        ))
        
        # Parse period
        days = int(period[:-1]) if period.endswith('d') else 30
        
        report = await privacy_manager.generate_report(
            report_id=report_id,
            period_start=datetime.now() - timedelta(days=days),
            period_end=datetime.now(),
            format=format
        )
        
        click.echo(f"Report generated: {report}")
    
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
            jurisdiction="global",
            regulations=["GDPR"]
        ))
        
        result = await privacy_manager.export_user_data(
            user_id=user_id,
            request_id=request_id,
            format=format
        )
        
        click.echo(f"Data exported: {result}")
    
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
            jurisdiction="global",
            regulations=["GDPR"]
        ))
        
        result = await privacy_manager.erase_user_data(
            user_id=user_id,
            request_id=request_id,
            verification_required=verify
        )
        
        click.echo(f"Data erased: {result}")
    
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
            jurisdiction="global",
            regulations=["GDPR", "AI_ACT"]
        ))
        
        # Parse metadata if provided
        metadata_dict = {}
        if metadata:
            import json
            metadata_dict = json.loads(metadata)
        
        result = await privacy_manager.request_model_approval(
            model_id=model_id,
            approver_email=approver,
            metadata=metadata_dict
        )
        
        click.echo(f"Model approval requested: {result}")
    
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
            jurisdiction="global",
            regulations=["GDPR", "AI_ACT"]
        ))
        
        # Parse security checks
        check_list = checks.split(',') if checks else ["dependency_scan", "license_check", "cve_lookup"]
        
        result = await privacy_manager.register_plugin(
            name=name,
            source=source,
            checks={check: True for check in check_list}
        )
        
        click.echo(f"Plugin registered: {result}")
    
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
            jurisdiction="global",
            regulations=["GDPR", "AI_ACT"]
        ))
        
        result = await privacy_manager.run_compliance_tests(
            suite_id=suite,
            integration={
                "ticket_system": ticket_system,
                "project": project
            } if ticket_system and project else None
        )
        
        click.echo(f"Test suite results: {result}")
    
    asyncio.run(_run_tests())

@governance.command()
@click.option('--store', required=True, help='Embedding store identifier')
@click.option('--threshold', type=float, default=0.15, help='Drift threshold')
def drift_check(store, threshold):
    """Check for embedding drift."""
    
    async def _check_drift():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            jurisdiction="global",
            regulations=["GDPR", "AI_ACT"]
        ))
        
        result = await privacy_manager.check_embedding_drift(
            store_id=store,
            threshold=threshold
        )
        
        click.echo(f"Drift check results: {result}")
    
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
            jurisdiction="global",
            regulations=["GDPR", "AI_ACT"]
        ))
        
        result = await privacy_manager.override_risk_score(
            request_id=request_id,
            new_score=new_score,
            reason=reason,
            officer_id=officer_id
        )
        
        click.echo(f"Risk score overridden: {result}")
    
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
            jurisdiction="global",
            regulations=["GDPR", "AI_ACT"]
        ))
        
        # Parse timestamps if provided
        start = datetime.fromisoformat(start_time) if start_time else None
        end = datetime.fromisoformat(end_time) if end_time else None
        
        result = await privacy_manager.verify_log_chain(
            chain_id=chain_id,
            start_time=start,
            end_time=end
        )
        
        click.echo(f"Log chain verification: {result}")
    
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
            jurisdiction="global",
            regulations=["GDPR", "AI_ACT"]
        ))
        
        # Parse metadata if provided
        metadata_dict = {}
        if metadata:
            import json
            metadata_dict = json.loads(metadata)
        
        result = await privacy_manager.publish_policy(
            policy_file=policy_file,
            version=version,
            metadata=metadata_dict
        )
        
        click.echo(f"Policy published: {result}")
    
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
            jurisdiction="global",
            regulations=["GDPR", "AI_ACT"]
        ))
        
        result = await privacy_manager.create_incident(
            type=type,
            details_file=details_file,
            severity=severity,
            playbook=playbook
        )
        
        click.echo(f"Incident created: {result}")
    
    asyncio.run(_create())

@governance.command()
@click.option('--days', type=int, default=7, help='Days until consent expiry')
@click.option('--channels', help='Comma-separated list of notification channels')
def consent_check(days, channels):
    """Check for expiring consents."""
    
    async def _check():
        privacy_manager = PrivacyCompliance(config=GovernanceConfig(
            organization_id="org_123",
            jurisdiction="global",
            regulations=["GDPR"]
        ))
        
        # Parse notification channels
        channel_list = channels.split(',') if channels else ["email"]
        
        result = await privacy_manager.check_consent_expiry(
            days_until_expiry=days,
            notification_channels=channel_list
        )
        
        click.echo(f"Expiring consents: {result}")
    
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
            jurisdiction="global",
            regulations=["GDPR", "AI_ACT"]
        ))
        
        result = await privacy_manager.assign_dpia_review(
            dataset_id=dataset_id,
            assignee=assignee,
            priority=priority,
            due_date=datetime.now() + timedelta(days=due_days)
        )
        
        click.echo(f"DPIA review assigned: {result}")
    
    asyncio.run(_assign())

if __name__ == '__main__':
    governance() 