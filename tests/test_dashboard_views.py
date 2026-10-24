"""Tests for the dashboard UI views (static bundle + the API shapes they rely on).

No network: the static index/app are read through TestClient, and the only
API exercised is the in-process /api/scan-text.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="requires multimind-sdk[gateway]")
from fastapi.testclient import TestClient

from multimind.dashboard.server import DashboardSettings, create_dashboard_app

STATIC_DIR = Path(__file__).parent.parent / "multimind" / "dashboard" / "static"

NEW_VIEWS = ("scan", "evidence", "compliance")
GOVERNANCE_VIEWS = ("overview", "audit", "costs", "inventory", "guardrails") + NEW_VIEWS
PLAYGROUND_VIEWS = (
    "features",
    "models",
    "chat",
    "compare",
    "rag",
    "hallucination",
    "agents",
    "memory",
    "vectorstores",
    "finetune",
    "conversion",
    "regulatory",
    "dp",
    "mcp",
    "integrations",
    "tracing",
    "context",
)
ALL_VIEWS = GOVERNANCE_VIEWS + PLAYGROUND_VIEWS + ("settings",)


@pytest.fixture()
def client(tmp_path):
    settings = DashboardSettings(
        audit_log=str(tmp_path / "audit.jsonl"),
        costs_log=str(tmp_path / "costs.jsonl"),
        project_path=str(tmp_path),
        guardrails_path=str(tmp_path / "guardrails.json"),
    )
    return TestClient(create_dashboard_app(settings))


@pytest.fixture(scope="module")
def app_js():
    return (STATIC_DIR / "app.js").read_text(encoding="utf-8")


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


def test_index_groups_nav_into_playground_governance_settings(client):
    html = client.get("/").text
    groups = re.findall(r'<div class="nav-group">(.*?)</div>', html)
    assert groups == ["Playground", "Governance", "Settings"]
    # Order: playground views first, governance second, settings last.
    assert html.index('data-view="features"') < html.index('data-view="overview"') < html.index(
        'data-view="settings"'
    )


def test_app_js_routes_every_nav_entry(client):
    app_js = client.get("/app.js").text
    routes = re.search(r"const routes = \{(.*?)\};", app_js, re.S).group(1)
    for name in ALL_VIEWS:
        assert re.search(rf"\b{name}:", routes), f"no route for {name}"
    # The Playground features table is the landing view.
    assert 'const DEFAULT_VIEW = "features"' in app_js
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


def test_app_js_references_every_feature_route(app_js):
    """Every route in GET /v1/features must be called from the bundle (prefix before {param})."""
    from multimind.gateway.features_api import _FEATURES

    missing = []
    for feature in _FEATURES:
        for route in feature["routes"]:
            path = route.split(" ", 1)[1]
            prefix = path.split("{", 1)[0].rstrip("/") or path
            if prefix not in app_js:
                missing.append(route)
    assert not missing, f"app.js never calls: {missing}"


def test_app_js_settings_view_uses_config_api_without_key_inputs(app_js):
    assert '"/api/config"' in app_js
    assert '"/api/config/test-provider"' in app_js
    assert "has_api_key" in app_js
    assert 'method: "PUT", json: changes' in app_js  # only changed keys are sent
    # No field ever edits a provider key.
    assert "pv-api-key" not in app_js
    assert re.search(r'class="pv-(enabled|model|base)"', app_js)
    assert "api_key" not in re.sub(r"has_api_key", "", app_js)


def test_app_js_connection_model(app_js):
    # Same-origin detection probes /v1/features on the page origin; 404 means standalone.
    assert 'fetch(API_BASE + "/v1/features"' in app_js
    assert 'resp.status === 404 ? "standalone" : "same-origin"' in app_js
    # The key comes from localStorage only; window.MULTIMIND_API_KEY is no longer consulted.
    assert "window.MULTIMIND_API_KEY" not in app_js
    assert 'localStorage.getItem("multimind.apiKey")' in app_js
    # Every status the task lists is explained to the user.
    for status in ("401", "404", "501", "status === 0"):
        assert status in app_js


def _extract_function(js: str, name: str) -> str:
    match = re.search(rf"^function {name}\([^)]*\) \{{.*?^\}}", js, re.S | re.M)
    assert match, f"function {name} not found"
    return match.group(0)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_api_key_header_attached_from_local_storage(app_js, tmp_path):
    helpers = _extract_function(app_js, "readApiKey") + "\n" + _extract_function(app_js, "authHeaders")
    script = (
        "globalThis.localStorage = { getItem: (k) => (k === 'multimind.apiKey' ? 'k-123' : null) };\n"
        + helpers
        + "\nconst h = authHeaders({ 'Content-Type': 'application/json' });\n"
        "if (h['X-API-Key'] !== 'k-123' || h['Content-Type'] !== 'application/json') process.exit(2);\n"
        "globalThis.localStorage = { getItem: () => null };\n"
        "if ('X-API-Key' in authHeaders()) process.exit(3);\n"
        "globalThis.localStorage = { getItem: () => { throw new Error('blocked'); } };\n"
        "if ('X-API-Key' in authHeaders()) process.exit(4);\n"
        "console.log('ok');\n"
    )
    path = tmp_path / "helpers.js"
    path.write_text(script, encoding="utf-8")
    result = subprocess.run(["node", str(path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_app_js_parses():
    result = subprocess.run(
        ["node", "--check", str(STATIC_DIR / "app.js")], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_static_bundle_has_no_absolute_urls():
    for path in (STATIC_DIR / "app.js", STATIC_DIR / "index.html"):
        text = path.read_text(encoding="utf-8")
        assert "http://" not in text and "https://" not in text, path.name


def test_platform_serves_dashboard_with_new_views(monkeypatch):
    from multimind.server.platform import create_platform_app

    monkeypatch.delenv("API_KEYS", raising=False)
    client = TestClient(create_platform_app())
    page = client.get("/")
    assert page.status_code == 200
    for name in ALL_VIEWS:
        assert f'data-view="{name}"' in page.text, name
    assert client.get("/app.js").status_code == 200
    assert client.get("/v1/features").json()["count"] >= 27


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
