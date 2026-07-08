"""Tests for the local AI governance dashboard (multimind.dashboard).

All endpoints are exercised over synthetic JSONL fixtures in tmp_path;
no live API calls, no network.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="requires multimind-sdk[gateway]")
from fastapi.testclient import TestClient

from multimind.dashboard.server import DashboardSettings, create_dashboard_app

STATIC_DIR = Path(__file__).parent.parent / "multimind" / "dashboard" / "static"


def _iso(**delta):
    return (datetime.now(timezone.utc) - timedelta(**delta)).isoformat()


AUDIT_RECORDS = [
    # oldest first, as an AuditLog file would be
    {
        "timestamp": _iso(days=3),
        "direction": "input",
        "method": "generate",
        "pii_types": {"email": 2},
        "count": 2,
        "strategy": "mask",
        "blocked": False,
        "tags": ["[EMAIL:aaaaaaaa]"],
    },
    {
        "timestamp": _iso(hours=2),
        "direction": "input",
        "method": "chat",
        "pii_types": {"ssn": 1},
        "count": 1,
        "strategy": "mask",
        "blocked": True,
        "tags": ["[SSN:bbbbbbbb]"],
    },
    {
        "timestamp": _iso(hours=1),
        "direction": "output",
        "method": "chat",
        "pii_types": {"email": 1, "phone": 1},
        "count": 2,
        "strategy": "mask",
        "blocked": False,
        "tags": [],
    },
]

COST_RECORDS = [
    {
        "provider": "openai",
        "model": "gpt-4o",
        "input_tokens": 100,
        "output_tokens": 50,
        "cost": 1.5,
        "tag": "team-a",
        "method": "chat",
        "estimated": False,
        "unpriced": False,
        "timestamp": _iso(days=2),
    },
    {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "input_tokens": 10,
        "output_tokens": 5,
        "cost": 0.5,
        "tag": "team-b",
        "method": "generate",
        "estimated": True,
        "unpriced": False,
        "timestamp": _iso(hours=3),
    },
    {
        "provider": "anthropic",
        "model": "claude-sonnet",
        "input_tokens": 20,
        "output_tokens": 10,
        "cost": 2.0,
        "tag": None,
        "method": "chat",
        "estimated": False,
        "unpriced": False,
        "timestamp": _iso(hours=1),
    },
]


@pytest.fixture()
def project_dir(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "requirements.txt").write_text("openai>=1.0\nrequests\n")
    (project / "main.py").write_text(
        'import os\nkey = os.environ["OPENAI_API_KEY"]\n'
        'leaked = "sk-' + "a1B2c3D4e5F6g7H8i9J0" + '"\n'
    )
    return project


@pytest.fixture()
def client(tmp_path, project_dir):
    audit_log = tmp_path / "audit.jsonl"
    audit_log.write_text("".join(json.dumps(r) + "\n" for r in AUDIT_RECORDS))
    costs_log = tmp_path / "costs.jsonl"
    costs_log.write_text("".join(json.dumps(r) + "\n" for r in COST_RECORDS))
    settings = DashboardSettings(
        audit_log=str(audit_log),
        costs_log=str(costs_log),
        project_path=str(project_dir),
        guardrails_path=str(tmp_path / "guardrails.json"),
    )
    return TestClient(create_dashboard_app(settings))


@pytest.fixture()
def empty_client(tmp_path):
    settings = DashboardSettings(
        audit_log=str(tmp_path / "missing-audit.jsonl"),
        costs_log=str(tmp_path / "missing-costs.jsonl"),
        project_path=str(tmp_path),
        guardrails_path=str(tmp_path / "guardrails.json"),
    )
    return TestClient(create_dashboard_app(settings))


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "healthy"
    assert client.get("/ready").json()["status"] == "ready"


def test_summary_math(client):
    data = client.get("/api/summary").json()
    audit = data["audit"]
    assert audit["available"] is True
    assert audit["all_time"]["events"] == 3
    assert audit["all_time"]["blocked"] == 1
    assert audit["all_time"]["pii_detections"] == 5
    assert audit["all_time"]["by_type"] == {"email": 3, "ssn": 1, "phone": 1}
    # the 3-day-old record falls outside the 24h window
    assert audit["last_24h"]["events"] == 2
    assert audit["last_24h"]["by_type"] == {"ssn": 1, "email": 1, "phone": 1}
    costs = data["costs"]
    assert costs["available"] is True
    assert costs["total_cost"] == pytest.approx(4.0)
    assert costs["last_24h_cost"] == pytest.approx(2.5)
    assert costs["calls"] == 3
    assert sorted(costs["models"]) == ["claude-sonnet", "gpt-4o", "gpt-4o-mini"]
    assert data["budget"] is None  # no guardrails file yet


def test_summary_budget_status(client, tmp_path):
    (tmp_path / "guardrails.json").write_text(
        json.dumps({"strategy": "mask", "block_on": [], "budget_max_cost": 3.0})
    )
    budget = client.get("/api/summary").json()["budget"]
    assert budget["max_cost"] == 3.0
    assert budget["spent"] == pytest.approx(4.0)
    assert budget["exceeded"] is True
    assert budget["remaining"] == 0.0


def test_audit_newest_first_and_pagination(client):
    data = client.get("/api/audit").json()
    assert data["total"] == 3
    assert [r["method"] for r in data["records"]] == ["chat", "chat", "generate"]
    assert data["records"][0]["direction"] == "output"  # newest record
    page = client.get("/api/audit?limit=1&offset=2").json()
    assert page["total"] == 3
    assert len(page["records"]) == 1
    assert page["records"][0]["method"] == "generate"  # oldest record


def test_audit_filters(client):
    by_type = client.get("/api/audit?type=email").json()
    assert by_type["total"] == 2
    assert all("email" in r["pii_types"] for r in by_type["records"])
    blocked = client.get("/api/audit?blocked=true").json()
    assert blocked["total"] == 1
    assert blocked["records"][0]["pii_types"] == {"ssn": 1}
    allowed = client.get("/api/audit?blocked=false").json()
    assert allowed["total"] == 2
    combo = client.get("/api/audit?type=email&blocked=true").json()
    assert combo["total"] == 0


def test_costs_chargeback_shape(client):
    data = client.get("/api/costs").json()
    assert data["available"] is True
    cb = data["chargeback"]
    assert cb["calls"] == 3
    assert cb["total_cost"] == pytest.approx(4.0)
    assert set(cb["by_tag"]) == {"team-a", "team-b", "(untagged)"}
    assert cb["by_tag"]["team-a"]["cost"] == pytest.approx(1.5)
    assert cb["by_tag"]["team-a"]["share_pct"] == pytest.approx(37.5)
    assert set(data["by_model"]) == {"gpt-4o", "gpt-4o-mini", "claude-sonnet"}
    assert data["by_model"]["gpt-4o"]["input_tokens"] == 100
    daily = data["daily"]
    assert [d["date"] for d in daily] == sorted(d["date"] for d in daily)
    assert sum(d["cost"] for d in daily) == pytest.approx(4.0)
    assert sum(d["calls"] for d in daily) == 3
    # the two same-day records are bucketed together
    assert daily[-1]["calls"] == 2
    assert daily[-1]["cost"] == pytest.approx(2.5)


def test_inventory_scan_and_cache(client, project_dir):
    data = client.get("/api/inventory").json()
    kinds = {(f["kind"], f["provider"]) for f in data["findings"]}
    assert ("dependency", "openai") in kinds
    assert ("env_read", "openai") in kinds
    assert ("hardcoded_key", "openai") in kinds
    assert len(data["risks"]["hardcoded_keys"]) == 1
    assert data["summary"]["scanned_files"] == 2
    # cached: a new file is invisible until refresh=true
    (project_dir / "extra.py").write_text("import anthropic\nurl = 'api.anthropic.com'\n")
    again = client.get("/api/inventory").json()
    assert again["scanned_at"] == data["scanned_at"]
    assert again["summary"]["scanned_files"] == 2
    fresh = client.get("/api/inventory?refresh=true").json()
    assert fresh["summary"]["scanned_files"] == 3
    assert ("api_url", "anthropic") in {(f["kind"], f["provider"]) for f in fresh["findings"]}


def test_guardrails_roundtrip(client, tmp_path):
    initial = client.get("/api/guardrails").json()
    assert initial["exists"] is False
    assert initial["config"] == {
        "strategy": "mask",
        "block_on": [],
        "budget_max_cost": None,
        "scan_output": True,
    }
    assert len(initial["pii_types"]) == 9
    payload = {
        "strategy": "hash",
        "block_on": ["ssn", "credit_card"],
        "budget_max_cost": 25.0,
        "scan_output": False,
    }
    saved = client.put("/api/guardrails", json=payload)
    assert saved.status_code == 200
    assert saved.json()["config"] == payload
    assert "--config" in saved.json()["apply_hint"]
    on_disk = json.loads((tmp_path / "guardrails.json").read_text())
    assert on_disk == payload
    assert client.get("/api/guardrails").json()["config"] == payload


@pytest.mark.parametrize(
    "payload",
    [
        {"strategy": "shred"},
        {"block_on": ["email", "not_a_pii_type"]},
        {"budget_max_cost": -5},
        {"strategy": "mask", "unknown_field": 1},
    ],
)
def test_guardrails_validation_errors(client, payload):
    resp = client.put("/api/guardrails", json=payload)
    assert resp.status_code == 422


def test_proxy_settings_from_file(tmp_path, monkeypatch):
    from multimind.gateway.guard_proxy import ProxySettings

    path = tmp_path / "guardrails.json"
    path.write_text(
        json.dumps(
            {
                "strategy": "hash",
                "block_on": ["ssn", "api_key"],
                "budget_max_cost": 10.0,
                "scan_output": False,
            }
        )
    )
    settings = ProxySettings.from_file(str(path))
    assert settings.strategy == "hash"
    assert settings.block_on == ("ssn", "api_key")
    assert settings.budget == 10.0
    assert settings.scan_output is False
    # keyword overrides beat file values
    assert ProxySettings.from_file(str(path), strategy="remove").strategy == "remove"


def test_serve_cli_accepts_config_flag():
    from click.testing import CliRunner

    from multimind.cli.serve import serve

    result = CliRunner().invoke(serve, ["--help"])
    assert result.exit_code == 0
    assert "--config" in result.output


def test_dashboard_cli_help():
    from click.testing import CliRunner

    from multimind.cli.dashboard import dashboard

    result = CliRunner().invoke(dashboard, ["--help"])
    assert result.exit_code == 0
    assert "--audit-log" in result.output
    assert "--project" in result.output


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("MULTIMIND_DASHBOARD_AUDIT_LOG", "/tmp/env-audit.jsonl")
    monkeypatch.setenv("MULTIMIND_DASHBOARD_PORT", "9000")
    settings = DashboardSettings.from_env(costs_log="/tmp/override.jsonl")
    assert settings.audit_log == "/tmp/env-audit.jsonl"
    assert settings.port == 9000
    assert settings.costs_log == "/tmp/override.jsonl"
    assert settings.host == "127.0.0.1"


def test_missing_artifacts_are_graceful(empty_client):
    summary = empty_client.get("/api/summary").json()
    assert summary["audit"]["available"] is False
    assert summary["audit"]["all_time"]["events"] == 0
    assert summary["costs"]["available"] is False
    assert summary["costs"]["total_cost"] == 0.0
    assert empty_client.get("/api/audit").json()["total"] == 0
    costs = empty_client.get("/api/costs").json()
    assert costs["available"] is False
    assert costs["daily"] == []


def test_scan_text_finds_email(client):
    text = "Contact alice@example.com or bob@example.org please"
    resp = client.post("/api/scan-text", json={"text": text})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["counts"] == {"email": 2}
    assert data["redacted"] is None
    first = data["findings"][0]
    assert first["type"] == "email"
    assert first["text"] == "alice@example.com"
    assert text[first["start"] : first["end"]] == first["text"]


def test_scan_text_mask_strategy_returns_redacted(client):
    resp = client.post(
        "/api/scan-text", json={"text": "mail alice@example.com", "strategy": "mask"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["redacted"] == "mail [EMAIL]"
    assert data["counts"] == {"email": 1}


def test_scan_text_empty_text_rejected(client):
    assert client.post("/api/scan-text", json={"text": ""}).status_code == 422
    assert client.post("/api/scan-text", json={}).status_code == 422


def test_scan_text_bad_strategy_rejected(client):
    resp = client.post("/api/scan-text", json={"text": "hello", "strategy": "shred"})
    assert resp.status_code == 422


def test_evidence_report_html(client):
    resp = client.get("/api/evidence-report")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/html; charset=utf-8"
    assert resp.headers["content-disposition"] == 'attachment; filename="evidence-report.html"'
    assert "AI Compliance Evidence Report" in resp.text
    assert "not a legal compliance determination" in resp.text


def test_evidence_report_markdown(client):
    resp = client.get("/api/evidence-report?format=md")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/markdown; charset=utf-8"
    assert resp.headers["content-disposition"] == 'attachment; filename="evidence-report.md"'
    assert resp.text.startswith("# AI Compliance Evidence Report")


def test_evidence_report_bad_format(client):
    assert client.get("/api/evidence-report?format=pdf").status_code == 422


def test_evidence_report_missing_artifacts(empty_client):
    resp = empty_client.get("/api/evidence-report?format=md")
    assert resp.status_code == 200
    assert "Audit log not provided" in resp.text
    assert "Costs log not provided" in resp.text


def test_static_index_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "MultiMind" in resp.text
    assert 'src="./app.js"' in resp.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200


def test_static_assets_are_local_only():
    files = sorted(STATIC_DIR.iterdir())
    assert files, "static assets missing"
    for path in files:
        content = path.read_text(encoding="utf-8")
        assert "http://" not in content, f"external reference in {path.name}"
        assert "https://" not in content, f"external reference in {path.name}"
