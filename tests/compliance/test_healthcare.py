"""Unit tests for multimind.compliance.healthcare rule evaluation and audit logging."""

import io
import json
from datetime import datetime, timedelta

import pytest

from multimind.compliance.governance import GovernanceConfig
from multimind.compliance.guard import AuditLog
from multimind.compliance.healthcare import HealthcareCompliance


def make_manager(**config_overrides):
    stream = io.StringIO()
    config = GovernanceConfig(
        organization_id="org-1",
        organization_name="Test Org",
        dpo_email="dpo@example.com",
        **config_overrides,
    )
    return HealthcareCompliance(config=config, audit_log=AuditLog(stream)), stream


def audit_events(stream):
    return [json.loads(line)["event"] for line in stream.getvalue().splitlines()]


@pytest.mark.asyncio
async def test_access_phi_records_access_log_and_audit():
    manager, stream = make_manager()
    await manager.process_phi("d1", "p1", {"note": "x"}, "record")
    result = await manager.access_phi("d1", "user-1", "treatment")

    assert result is not None
    assert result.access_count == 1
    log = await manager.get_phi_access_log(data_id="d1")
    assert len(log) == 1
    assert log[0]["user_id"] == "user-1"
    assert log[0]["patient_id"] == "p1"
    assert "phi_access" in audit_events(stream)


@pytest.mark.asyncio
async def test_get_phi_access_log_filters():
    manager, _ = make_manager()
    await manager.process_phi("d1", "p1", "a", "record")
    await manager.process_phi("d2", "p2", "b", "record")
    await manager.access_phi("d1", "u1", "treatment")
    await manager.access_phi("d2", "u2", "billing")

    assert len(await manager.get_phi_access_log()) == 2
    assert len(await manager.get_phi_access_log(patient_id="p2")) == 1
    future = datetime.now() + timedelta(hours=1)
    assert await manager.get_phi_access_log(start_time=future) == []


@pytest.mark.asyncio
async def test_hipaa_validation_clean_state_is_unverifiable_not_compliant():
    manager, stream = make_manager()
    validation = await manager.validate_hipaa_compliance("sys-1")

    # Enforcement Rule cannot be verified from SDK state, so the overall
    # verdict must not claim full compliance.
    assert validation["status"] == "unverifiable"
    by_req = {r["requirement"]: r for r in validation["requirements"]}
    assert by_req["privacy_rule"]["status"] == "compliant"
    assert by_req["security_rule"]["status"] == "compliant"
    assert by_req["breach_notification"]["status"] == "compliant"
    assert by_req["data_retention"]["status"] == "compliant"
    assert by_req["enforcement_rule"]["status"] == "unverifiable"
    assert "hipaa_validation" in audit_events(stream)


@pytest.mark.asyncio
async def test_hipaa_validation_flags_disabled_safeguards():
    manager, _ = make_manager(enable_encryption=False, enable_audit_logging=False)
    validation = await manager.validate_hipaa_compliance("sys-1")

    assert validation["status"] == "non_compliant"
    by_req = {r["requirement"]: r for r in validation["requirements"]}
    assert by_req["security_rule"]["status"] == "non_compliant"
    assert "encryption" in by_req["security_rule"]["details"]
    assert by_req["privacy_rule"]["status"] == "non_compliant"


@pytest.mark.asyncio
async def test_hipaa_validation_flags_unlogged_access():
    manager, _ = make_manager()
    phi = await manager.process_phi("d1", "p1", "a", "record")
    # Simulate an access that bypassed the audit trail.
    phi.access_count = 3
    validation = await manager.validate_hipaa_compliance("sys-1")

    by_req = {r["requirement"]: r for r in validation["requirements"]}
    assert by_req["privacy_rule"]["status"] == "non_compliant"
    assert "d1" in by_req["privacy_rule"]["details"]


@pytest.mark.asyncio
async def test_hipaa_validation_flags_expired_retention():
    manager, _ = make_manager(data_retention_days=30)
    phi = await manager.process_phi("old", "p1", "a", "record")
    phi.created_at = datetime.now() - timedelta(days=45)
    validation = await manager.validate_hipaa_compliance("sys-1")

    by_req = {r["requirement"]: r for r in validation["requirements"]}
    assert by_req["data_retention"]["status"] == "non_compliant"
    assert "old" in by_req["data_retention"]["details"]


@pytest.mark.asyncio
async def test_breach_notification_check():
    manager, stream = make_manager()
    breach = await manager.report_breach("unauthorized_access", ["d1"], "leak", "critical")
    assert breach["notification_logged_at"] is not None
    assert "breach_notification" in audit_events(stream)

    validation = await manager.validate_hipaa_compliance("sys-1")
    by_req = {r["requirement"]: r for r in validation["requirements"]}
    assert by_req["breach_notification"]["status"] == "compliant"

    # A high-severity breach with no notification logged must fail the check.
    manager.breach_log.append(
        {
            "breach_id": "breach_manual",
            "timestamp": datetime.now(),
            "breach_type": "loss",
            "affected_data": ["d2"],
            "description": "lost device",
            "severity": "high",
            "status": "reported",
            "resolution": None,
            "resolved_at": None,
            "notification_logged_at": None,
        }
    )
    validation = await manager.validate_hipaa_compliance("sys-1")
    by_req = {r["requirement"]: r for r in validation["requirements"]}
    assert by_req["breach_notification"]["status"] == "non_compliant"
    assert "breach_manual" in by_req["breach_notification"]["details"]


@pytest.mark.asyncio
async def test_breach_unresolved_past_deadline_is_non_compliant():
    manager, _ = make_manager()
    breach = await manager.report_breach("unauthorized_access", ["d1"], "leak", "low")
    breach["timestamp"] = datetime.now() - timedelta(days=90)

    validation = await manager.validate_hipaa_compliance("sys-1")
    by_req = {r["requirement"]: r for r in validation["requirements"]}
    assert by_req["breach_notification"]["status"] == "non_compliant"
    assert "60-day" in by_req["breach_notification"]["details"]


@pytest.mark.asyncio
async def test_hitech_validation_reports_unverifiable_requirements():
    manager, stream = make_manager()
    validation = await manager.validate_hitech_compliance("sys-1")

    assert validation["status"] == "unverifiable"
    by_req = {r["requirement"]: r for r in validation["requirements"]}
    assert by_req["meaningful_use"]["status"] == "unverifiable"
    assert by_req["health_information_exchange"]["status"] == "unverifiable"
    assert by_req["breach_notification"]["status"] == "compliant"
    assert "hitech_validation" in audit_events(stream)


@pytest.mark.asyncio
async def test_works_without_audit_log():
    config = GovernanceConfig(
        organization_id="org-1", organization_name="Test Org", dpo_email="dpo@example.com"
    )
    manager = HealthcareCompliance(config=config)
    await manager.process_phi("d1", "p1", "a", "record")
    await manager.access_phi("d1", "u1", "treatment")
    validation = await manager.validate_hipaa_compliance("sys-1")
    assert validation["status"] == "unverifiable"
