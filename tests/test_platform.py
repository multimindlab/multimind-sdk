"""The stable API serves one OpenAPI document."""

import pytest

pytest.importorskip("fastapi", reason="requires multimind-sdk[gateway]")
from fastapi.testclient import TestClient

from multimind.server.platform import create_platform_app


@pytest.fixture(autouse=True)
def _fresh_compliance_state():
    from multimind.gateway.compliance_api import reset_compliance_state

    reset_compliance_state()
    yield
    reset_compliance_state()


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    return TestClient(create_platform_app())


def test_single_openapi_includes_supported_routes(client):
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/v1/chat" in paths
    assert "/v1/sessions/{session_id}/messages" in paths
    assert "/v1/compliance/monitor" in paths
    assert "/v1/compliance/example/{type}" not in paths
    assert "/api/audit" in paths
    assert "/v1/chat/completions" in paths
    assert "/v1/guardrails/reload" in paths
    assert "/v1/process" not in paths
    session = paths["/v1/sessions/{session_id}"]
    assert "get" in session and "delete" in session


def test_api_key_required_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret")
    client = TestClient(create_platform_app())
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 401
    ok = client.get("/v1/models", headers={"X-API-Key": "secret"})
    assert ok.status_code != 401


def test_cors_allows_the_dashboard_origin(monkeypatch):
    monkeypatch.setenv("MULTIMIND_CORS_ORIGINS", "http://localhost:8501")
    monkeypatch.setenv("API_KEYS", "secret")
    client = TestClient(create_platform_app())
    response = client.get("/health", headers={"Origin": "http://localhost:8501"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:8501"
    preflight = client.options(
        "/api/audit",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-api-key",
        },
    )
    assert preflight.status_code != 401
    assert preflight.headers.get("access-control-allow-origin") == "http://localhost:8501"


def test_compliance_dashboard_stays_empty_without_recorded_checks(client):
    alerts = client.get("/v1/compliance/alerts", params={"organization_id": "org"})
    assert alerts.status_code == 200
    assert alerts.json() == []
    dashboard = client.get("/v1/compliance/dashboard", params={"organization_id": "org"})
    assert dashboard.status_code == 200
    body = dashboard.json()
    assert body["total_checks"] == 0
    assert body["compliance_score"] == 0
    configure = client.post(
        "/v1/compliance/alerts/configure",
        params={"organization_id": "org"},
        json={"rules": []},
    )
    assert configure.status_code == 200


def test_compliance_monitor_is_not_a_fabricated_pass(client):
    response = client.post(
        "/v1/compliance/monitor",
        json={
            "organization_id": "org",
            "organization_name": "Org",
            "dpo_email": "dpo@example.com",
            "enabled_regulations": ["GDPR"],
            "compliance_rules": {},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["final_evaluation"]["status"] == "unverifiable"
    assert body["metrics"] == {}
    dashboard = client.get("/v1/compliance/dashboard", params={"organization_id": "org"}).json()
    assert dashboard["total_checks"] == 1
    assert dashboard["unverifiable_checks"] == 1
    assert dashboard["passed_checks"] == 0 and dashboard["failed_checks"] == 0


def test_compliance_dashboard_and_alerts_reflect_recorded_checks(client):
    config = {
        "organization_id": "org",
        "organization_name": "Org",
        "dpo_email": "dpo@example.com",
        "enabled_regulations": [],
        "compliance_rules": {},
    }
    clean = client.post(
        "/v1/compliance/monitor", json={**config, "metadata": {"text": "hello world"}}
    )
    assert clean.json()["final_evaluation"]["status"] == "checked"
    pii = client.post(
        "/v1/compliance/monitor",
        json={**config, "metadata": {"text": "mail me at jane@example.com"}},
    )
    assert pii.json()["final_evaluation"]["status"] == "attention"

    dashboard = client.get("/v1/compliance/dashboard", params={"organization_id": "org"}).json()
    assert dashboard["total_checks"] == 2
    assert dashboard["passed_checks"] == 1
    assert dashboard["failed_checks"] == 1
    assert dashboard["compliance_score"] == 0.5
    assert dashboard["trend_data"] == {"compliance_score": [1.0, 0.0]}
    assert len(dashboard["recent_issues"]) == 1
    assert "pii_scan: findings" in dashboard["recent_issues"][0]["description"]

    alerts = client.get("/v1/compliance/alerts", params={"organization_id": "org"}).json()
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "high" and alerts[0]["status"] == "active"
    assert client.get("/v1/compliance/alerts", params={"organization_id": "other"}).json() == []
