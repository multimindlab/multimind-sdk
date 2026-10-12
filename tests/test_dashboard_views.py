"""Tests for the dashboard UI views (static bundle + the API shapes they rely on).

No network: the static index/app are read through TestClient, and the only
API exercised is the in-process /api/scan-text.
"""

import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="requires multimind-sdk[gateway]")
from fastapi.testclient import TestClient

from multimind.dashboard.server import DashboardSettings, create_dashboard_app

STATIC_DIR = Path(__file__).parent.parent / "multimind" / "dashboard" / "static"

NEW_VIEWS = ("scan", "evidence", "compliance")
ALL_VIEWS = ("overview", "audit", "costs", "inventory", "guardrails") + NEW_VIEWS


@pytest.fixture()
def client(tmp_path):
    settings = DashboardSettings(
        audit_log=str(tmp_path / "audit.jsonl"),
        costs_log=str(tmp_path / "costs.jsonl"),
        project_path=str(tmp_path),
        guardrails_path=str(tmp_path / "guardrails.json"),
    )
    return TestClient(create_dashboard_app(settings))


def test_index_has_nav_entries_for_every_view(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    for name in ALL_VIEWS:
        assert f'href="#/{name}"' in html, name
        assert f'data-view="{name}"' in html, name
    # Sidebar labels for the new views.
    assert "Scan Text" in html
    assert "Evidence Report" in html
    assert "Compliance" in html


def test_app_js_routes_every_nav_entry(client):
    app_js = client.get("/app.js").text
    routes = re.search(r"const routes = \{(.*?)\};", app_js, re.S).group(1)
    for name in ALL_VIEWS:
        assert re.search(rf"\b{name}:", routes), f"no route for {name}"
    # Each view renders the element ids the tests below and the UI rely on.
    for element_id in ("scan-form", "scan-text", "scan-strategy", "scan-result"):
        assert f'id="{element_id}"' in app_js
    for element_id in ("gr-proxy-url", "gr-reload", "gr-reload-msg", "gr-hint"):
        assert f'id="{element_id}"' in app_js
    for element_id in ("cp-base", "cp-org", "cp-form", "cp-text", "cp-metrics"):
        assert f'id="{element_id}"' in app_js
    assert 'id="ev-${f}"' in app_js  # one download link per report format


def test_app_js_calls_expected_endpoints(client):
    app_js = client.get("/app.js").text
    assert '"/api/scan-text"' in app_js
    assert "/api/evidence-report?format=" in app_js
    assert '"/v1/guardrails/reload"' in app_js
    assert "/v1/compliance/dashboard?" in app_js
    assert "/v1/compliance/alerts?" in app_js
    assert '"/v1/compliance/monitor"' in app_js
    # Saved guardrails surface the server's apply_hint.
    assert "apply_hint" in app_js
    # Backend/proxy URLs and the org id are remembered per browser.
    for key in ("multimind.proxyBase", "multimind.backendBase", "multimind.orgId"):
        assert key in app_js
    # Unreachable backend must point the user at `multimind backend`.
    assert "multimind backend" in app_js


def test_static_bundle_has_no_absolute_urls():
    for path in (STATIC_DIR / "app.js", STATIC_DIR / "index.html"):
        text = path.read_text(encoding="utf-8")
        assert "http://" not in text and "https://" not in text, path.name


def test_scan_text_shape_matches_ui(client):
    text = "Email bob@example.org and SSN 123-45-6789"
    resp = client.post("/api/scan-text", json={"text": text, "strategy": "mask"})
    assert resp.status_code == 200
    data = resp.json()
    # Keys the Scan Text view reads.
    assert set(data) == {"findings", "counts", "total", "redacted"}
    assert data["total"] == len(data["findings"]) >= 2
    for finding in data["findings"]:
        assert set(finding) == {"type", "start", "end", "text"}
        assert text[finding["start"] : finding["end"]] == finding["text"]
    assert set(data["counts"]) == {f["type"] for f in data["findings"]}
    assert isinstance(data["redacted"], str)
    assert "bob@example.org" not in data["redacted"]
    # Detect-only mode: the UI hides the redacted card when it is null.
    detect = client.post("/api/scan-text", json={"text": text}).json()
    assert detect["redacted"] is None
    assert detect["total"] == data["total"]


def test_scan_text_rejects_unknown_strategy(client):
    resp = client.post("/api/scan-text", json={"text": "hi", "strategy": "shred"})
    assert resp.status_code == 422
    # The UI shows detail[0].msg from this payload.
    assert resp.json()["detail"][0]["msg"]


def test_guardrails_put_returns_apply_hint_and_path(client):
    payload = {"strategy": "mask", "block_on": ["ssn"], "budget_max_cost": None, "scan_output": True}
    data = client.put("/api/guardrails", json=payload).json()
    assert data["path"].endswith("guardrails.json")
    assert "/v1/guardrails/reload" in data["apply_hint"]


def test_evidence_report_links_resolve(client):
    for fmt, ctype in (("html", "text/html"), ("md", "text/markdown")):
        resp = client.get(f"/api/evidence-report?format={fmt}")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith(ctype)
