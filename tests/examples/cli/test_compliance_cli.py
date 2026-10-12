"""
Tests for compliance_cli.py CLI example.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

# Add root directory to path
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

from examples.cli.compliance_cli import governance
from multimind.compliance.governance import DataCategory, Regulation
from multimind.compliance.privacy import (
    AuditAction,
    AuditTrail,
    ComplianceEvent,
    ComplianceReport,
    ComplianceReportTemplate,
    ComplianceScore,
    ComplianceWorkflow,
    DataPurpose,
    GovernanceConfig,
    NotificationType,
    PrivacyCompliance,
    PrivacyData,
    RiskScore,
)


class MockPrivacyCompliance:
    """Mock PrivacyCompliance for testing."""

    def __init__(self, config=None):
        self.config = config or GovernanceConfig(
            organization_id="org_123",
            organization_name="Example Organization",
            dpo_email="dpo@example.com",
            enabled_regulations=[Regulation.GDPR, Regulation.AI_ACT]
        )
        self.privacy_data = {}
        self.data_purposes = {}
        self.report_templates = {}
        self.risk_scores = {}
        self.workflows = {}
        self.notifications = []
        self.compliance_calendar = {}
        self.audit_trails = []
        self.anomalies = []
        self.compliance_reports = []
        self.consent_history = []

    async def add_data_purpose(
        self,
        purpose_id: str,
        name: str,
        description: str,
        legal_basis: str,
        retention_period: int,
        data_categories: set
    ):
        purpose = DataPurpose(
            purpose_id=purpose_id,
            name=name,
            description=description,
            legal_basis=legal_basis,
            retention_period=retention_period,
            data_categories=data_categories
        )
        self.data_purposes[purpose_id] = purpose
        return purpose

    async def process_privacy_data(
        self,
        data_id: str,
        data_type: str,
        content: dict,
        jurisdiction: str,
        data_categories: set,
        purposes: set,
        metadata: dict = None
    ):
        privacy_data = PrivacyData(
            data_id=data_id,
            data_type=data_type,
            content=content,
            jurisdiction=jurisdiction,
            data_categories=data_categories,
            purposes=purposes,
            metadata=metadata or {},
            retention_end_date=datetime.now() + timedelta(days=365)
        )
        self.privacy_data[data_id] = privacy_data
        return privacy_data

    async def get_audit_trails(
        self,
        entity_id: str = None,
        start_date: datetime = None,
        end_date: datetime = None
    ):
        trails = self.audit_trails
        if entity_id:
            trails = [t for t in trails if t.entity_id == entity_id]
        if start_date:
            trails = [t for t in trails if t.timestamp >= start_date]
        if end_date:
            trails = [t for t in trails if t.timestamp <= end_date]
        return trails

    async def generate_compliance_report(
        self,
        template_id: str,
        period_start: datetime,
        period_end: datetime,
        jurisdiction: str,
        regulation: str,
        metadata: dict = None
    ):
        report = ComplianceReport(
            report_id=f"report_{datetime.now().timestamp()}",
            template_id=template_id,
            period_start=period_start,
            period_end=period_end,
            jurisdiction=jurisdiction,
            regulation=regulation,
            overall_status="compliant",
            findings=[],
            recommendations=[],
            metadata=metadata or {}
        )
        self.compliance_reports.append(report)
        return report

    async def export_data_portability(self, user_id: str, format: str = "json"):
        return f'{{"user_id": "{user_id}", "data": "exported"}}'

    async def process_data_subject_request(
        self,
        request_type: str,
        user_id: str,
        data_ids: list,
        metadata: dict = None
    ):
        return {
            "request_type": request_type,
            "user_id": user_id,
            "data_ids": data_ids,
            "status": "completed"
        }

    async def create_compliance_workflow(
        self,
        workflow_id: str,
        name: str,
        description: str,
        steps: list,
        assigned_to: str = None,
        metadata: dict = None
    ):
        workflow = ComplianceWorkflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            steps=steps,
            current_step=0,
            assigned_to=assigned_to,
            metadata=metadata or {}
        )
        self.workflows[workflow_id] = workflow
        return workflow

    async def create_audit_trail(
        self,
        action: AuditAction,
        entity_type: str,
        entity_id: str,
        user_id: str,
        metadata: dict = None
    ):
        trail = AuditTrail(
            trail_id=f"trail_{datetime.now().timestamp()}",
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=user_id,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.audit_trails.append(trail)
        return trail

    async def detect_anomalies(self):
        return self.anomalies

    async def calculate_risk_score(self, entity_id: str):
        if entity_id in self.risk_scores:
            return self.risk_scores[entity_id]
        return RiskScore(score=0.5, level="medium", factors=[])

    async def calculate_compliance_score(
        self,
        entity_id: str,
        regulation: str,
        jurisdiction: str
    ):
        return ComplianceScore(
            score_id=f"score_{entity_id}",
            entity_id=entity_id,
            regulation=regulation,
            jurisdiction=jurisdiction,
            score=85.0,
            trend="improving",
            components={"data_protection": 0.9, "privacy": 0.8}
        )

    async def create_compliance_event(
        self,
        title: str,
        description: str,
        event_type: str,
        start_date: datetime,
        end_date: datetime = None,
        jurisdiction: str = "global",
        regulation: str = "general",
        assigned_to: str = None
    ):
        event = ComplianceEvent(
            event_id=f"event_{datetime.now().timestamp()}",
            title=title,
            description=description,
            event_type=event_type,
            start_date=start_date,
            end_date=end_date,
            jurisdiction=jurisdiction,
            regulation=regulation,
            assigned_to=assigned_to
        )
        self.compliance_calendar[event.event_id] = event
        return event

    async def create_notification(
        self,
        type: NotificationType,
        title: str,
        message: str,
        priority: str,
        recipient: str,
        metadata: dict = None
    ):
        notification = {
            "type": type,
            "title": title,
            "message": message,
            "priority": priority,
            "recipient": recipient,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        }
        self.notifications.append(notification)
        return notification

    async def get_consent_history(self):
        return self.consent_history


@pytest.fixture
def mock_privacy_compliance():
    """Fixture to provide mock PrivacyCompliance."""
    return MockPrivacyCompliance()


@pytest.fixture
def cli_runner():
    """Fixture to provide Click CLI runner."""
    return CliRunner()


@pytest.mark.asyncio
async def test_compliance_cli_imports():
    """Test that the compliance_cli module can be imported."""
    try:
        from examples.cli import compliance_cli
        assert compliance_cli is not None
        assert hasattr(compliance_cli, 'governance')
    except ImportError as e:
        pytest.fail(f"Failed to import compliance_cli: {e}")


def test_governance_group_exists(cli_runner):
    """Test that the governance CLI group exists."""
    result = cli_runner.invoke(governance, ['--help'])
    assert result.exit_code == 0
    assert "MultiMind Governance CLI" in result.output


def test_ingest_command(cli_runner, mock_privacy_compliance):
    """Test the ingest command."""
    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'ingest',
            '--dataset-id', 'test_dataset_1',
            '--name', 'Test Dataset',
            '--description', 'A test dataset',
            '--data-categories', 'personal,financial'
        ])
        assert result.exit_code == 0
        assert "Dataset ingested successfully" in result.output
        assert "test_dataset_1" in result.output


def test_validate_output_command(cli_runner, mock_privacy_compliance):
    """Test the validate-output command."""
    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'validate-output',
            '--output-id', 'output_1',
            '--content', 'Test content',
            '--user-id', 'user_123',
            '--purpose', 'testing'
        ])
        assert result.exit_code == 0
        assert "Validation completed" in result.output
        assert "output_1" in result.output


def test_monitor_anomalies_command(cli_runner, mock_privacy_compliance):
    """Test the monitor-anomalies command."""
    # Add some mock audit trails
    mock_privacy_compliance.audit_trails = [
        AuditTrail(
            trail_id="trail_1",
            action=AuditAction.CREATE,
            entity_type="data",
            entity_id="data_1",
            user_id="user_1",
            timestamp=datetime.now(),
            metadata={"severity": "HIGH"}
        )
    ]

    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        start_time = datetime.now().isoformat()
        result = cli_runner.invoke(governance, [
            'monitor-anomalies',
            '--start-time', start_time
        ])
        assert result.exit_code == 0


def test_export_logs_command(cli_runner, mock_privacy_compliance):
    """Test the export-logs command."""
    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'export-logs',
            '--report-id', 'report_1',
            '--period', '30d',
            '--format', 'pdf'
        ])
        assert result.exit_code == 0
        assert "Report generated successfully" in result.output


def test_dsar_export_command(cli_runner, mock_privacy_compliance):
    """Test the dsar export command."""
    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'dsar', 'export',
            '--user-id', 'user_123',
            '--request-id', 'req_1',
            '--format', 'json'
        ])
        assert result.exit_code == 0
        assert "DSAR export completed" in result.output
        assert "user_123" in result.output


def test_dsar_erase_command(cli_runner, mock_privacy_compliance):
    """Test the dsar erase command."""
    # Add some mock privacy data
    mock_privacy_compliance.privacy_data = {
        "data_1": PrivacyData(
            data_id="data_1",
            data_type="dataset",
            content={"name": "Test"},
            jurisdiction="global",
            data_categories={DataCategory.PERSONAL},
            purposes={"purpose_1"},  # At least one purpose required
            metadata={"user_id": "user_123"}
        )
    }

    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'dsar', 'erase',
            '--user-id', 'user_123',
            '--request-id', 'req_1',
            '--verify'
        ])
        assert result.exit_code == 0
        assert "DSAR erasure completed" in result.output


def test_model_approve_command(cli_runner, mock_privacy_compliance):
    """Test the model-approve command."""
    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'model-approve',
            '--model-id', 'model_1',
            '--approver', 'approver@example.com'
        ])
        assert result.exit_code == 0
        assert "Model approval workflow created" in result.output


def test_plugin_register_command(cli_runner, mock_privacy_compliance):
    """Test the plugin-register command."""
    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'plugin-register',
            '--name', 'test_plugin',
            '--source', 'https://example.com/plugin'
        ])
        assert result.exit_code == 0
        assert "Plugin registration recorded" in result.output


def test_test_run_command(cli_runner, mock_privacy_compliance):
    """Test the test-run command."""
    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'test-run',
            '--suite', 'test_suite_1'
        ])
        assert result.exit_code == 0
        assert "Compliance test summary" in result.output


def test_drift_check_command(cli_runner, mock_privacy_compliance):
    """Test the drift-check command."""
    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'drift-check',
            '--store', 'store_1',
            '--threshold', '0.15'
        ])
        assert result.exit_code == 0
        assert "Embedding drift check" in result.output


def test_risk_override_command(cli_runner, mock_privacy_compliance):
    """Test the risk-override command."""
    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'risk-override',
            '--request-id', 'req_1',
            '--new-score', '0.8',
            '--reason', 'Manual override',
            '--officer-id', 'officer_1'
        ])
        assert result.exit_code == 0
        assert "Risk score override recorded" in result.output


def test_audit_verify_command(cli_runner, mock_privacy_compliance):
    """Test the audit-verify command."""
    # Add some mock audit trails
    mock_privacy_compliance.audit_trails = [
        AuditTrail(
            trail_id="trail_1",
            action=AuditAction.CREATE,
            entity_type="data",
            entity_id="chain_1",
            user_id="user_1",
            timestamp=datetime.now(),
            metadata={}
        )
    ]

    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'audit-verify',
            '--chain-id', 'chain_1'
        ])
        assert result.exit_code == 0
        assert "Audit chain verification summary" in result.output


def test_policy_publish_command(cli_runner, mock_privacy_compliance, tmp_path):
    """Test the policy-publish command."""
    # Create a temporary policy file
    policy_file = tmp_path / "policy.txt"
    policy_file.write_text("Test policy content")

    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'policy-publish',
            '--policy-file', str(policy_file),
            '--version', '1.0'
        ])
        assert result.exit_code == 0
        assert "Policy publication recorded" in result.output


def test_incident_create_command(cli_runner, mock_privacy_compliance, tmp_path):
    """Test the incident-create command."""
    # Create a temporary incident details file
    details_file = tmp_path / "incident.txt"
    details_file.write_text("Test incident details")

    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'incident-create',
            '--type', 'data_breach',
            '--details-file', str(details_file),
            '--severity', 'high'
        ])
        assert result.exit_code == 0
        assert "Incident recorded and team notified" in result.output


def test_consent_check_command(cli_runner, mock_privacy_compliance):
    """Test the consent-check command."""
    # Add some mock consent history
    mock_privacy_compliance.consent_history = [
        {
            "consent_id": "consent_1",
            "user_id": "user_1",
            "timestamp": datetime.now() + timedelta(days=5),
            "granted": True
        }
    ]

    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'consent-check',
            '--days', '7'
        ])
        assert result.exit_code == 0
        assert "Consent check summary" in result.output


def test_dpia_assign_command(cli_runner, mock_privacy_compliance):
    """Test the dpia-assign command."""
    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'dpia-assign',
            '--dataset-id', 'dataset_1',
            '--assignee', 'assignee@example.com',
            '--priority', 'high',
            '--due-days', '14'
        ])
        assert result.exit_code == 0
        assert "DPIA review task created" in result.output


def test_ingest_command_with_metadata(cli_runner, mock_privacy_compliance):
    """Test the ingest command with metadata."""
    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'ingest',
            '--dataset-id', 'test_dataset_2',
            '--name', 'Test Dataset 2',
            '--description', 'Another test dataset',
            '--data-categories', 'personal',
            '--metadata', '{"key": "value"}'
        ])
        assert result.exit_code == 0
        assert "Dataset ingested successfully" in result.output


def test_export_logs_command_json_format(cli_runner, mock_privacy_compliance):
    """Test the export-logs command with JSON format."""
    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'export-logs',
            '--report-id', 'report_2',
            '--period', '60d',
            '--format', 'json'
        ])
        assert result.exit_code == 0
        assert "Report generated successfully" in result.output


def test_monitor_anomalies_with_severity_filter(cli_runner, mock_privacy_compliance):
    """Test monitor-anomalies with severity filter."""
    mock_privacy_compliance.audit_trails = [
        AuditTrail(
            trail_id="trail_1",
            action=AuditAction.CREATE,
            entity_type="data",
            entity_id="data_1",
            user_id="user_1",
            timestamp=datetime.now(),
            metadata={"severity": "HIGH"}
        )
    ]

    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        start_time = datetime.now().isoformat()
        result = cli_runner.invoke(governance, [
            'monitor-anomalies',
            '--start-time', start_time,
            '--severity', 'HIGH,CRITICAL'
        ])
        assert result.exit_code == 0


def test_test_run_with_ticket_system(cli_runner, mock_privacy_compliance):
    """Test test-run command with ticket system integration."""
    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'test-run',
            '--suite', 'test_suite_2',
            '--ticket-system', 'jira',
            '--project', 'PROJ-123'
        ])
        assert result.exit_code == 0
        assert "Compliance test summary" in result.output


def test_plugin_register_with_checks(cli_runner, mock_privacy_compliance):
    """Test plugin-register with security checks."""
    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'plugin-register',
            '--name', 'test_plugin_2',
            '--source', 'https://example.com/plugin2',
            '--checks', 'dependency_scan,license_check'
        ])
        assert result.exit_code == 0
        assert "Plugin registration recorded" in result.output


def test_consent_check_with_channels(cli_runner, mock_privacy_compliance):
    """Test consent-check with notification channels."""
    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'consent-check',
            '--days', '14',
            '--channels', 'email,sms'
        ])
        assert result.exit_code == 0
        assert "Consent check summary" in result.output


def test_incident_create_with_playbook(cli_runner, mock_privacy_compliance, tmp_path):
    """Test incident-create with playbook."""
    details_file = tmp_path / "incident.txt"
    details_file.write_text("Test incident details")

    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'incident-create',
            '--type', 'security_incident',
            '--details-file', str(details_file),
            '--severity', 'critical',
            '--playbook', 'security_response_playbook'
        ])
        assert result.exit_code == 0
        assert "Incident recorded and team notified" in result.output


def test_policy_publish_with_metadata(cli_runner, mock_privacy_compliance, tmp_path):
    """Test policy-publish with metadata."""
    policy_file = tmp_path / "policy.txt"
    policy_file.write_text("Test policy content")

    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'policy-publish',
            '--policy-file', str(policy_file),
            '--version', '2.0',
            '--metadata', '{"author": "test", "department": "compliance"}'
        ])
        assert result.exit_code == 0
        assert "Policy publication recorded" in result.output


def test_model_approve_with_metadata(cli_runner, mock_privacy_compliance):
    """Test model-approve with metadata."""
    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'model-approve',
            '--model-id', 'model_2',
            '--approver', 'approver2@example.com',
            '--metadata', '{"version": "1.0", "notes": "test"}'
        ])
        assert result.exit_code == 0
        assert "Model approval workflow created" in result.output


def test_error_handling_missing_file(cli_runner, mock_privacy_compliance):
    """Test error handling when file is missing."""
    with patch('examples.cli.compliance_cli.PrivacyCompliance', return_value=mock_privacy_compliance):
        result = cli_runner.invoke(governance, [
            'policy-publish',
            '--policy-file', '/nonexistent/policy.txt',
            '--version', '1.0'
        ])
        assert result.exit_code != 0 or "not found" in result.output.lower()


def test_error_handling_missing_required_options(cli_runner):
    """Test error handling when required options are missing."""
    result = cli_runner.invoke(governance, [
        'ingest',
        '--dataset-id', 'test_dataset'
        # Missing required options
    ])
    assert result.exit_code != 0


def test_compliance_cli_structure():
    """Test that the compliance CLI has the expected structure."""
    example_path = Path(__file__).parent.parent.parent.parent / "examples" / "cli" / "compliance_cli.py"
    assert example_path.exists(), "compliance_cli.py example should exist"

    # Check that the file contains expected components
    with open(example_path, 'r') as f:
        content = f.read()
        assert "@click.group()" in content
        assert "def governance()" in content
        assert "ingest" in content
        assert "validate_output" in content  # Function name uses underscore
        assert "dsar" in content

