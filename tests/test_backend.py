"""Tests for multimind.backend (the single-process REST API aggregator)."""

import pytest

pytest.importorskip("fastapi", reason="requires multimind-sdk[gateway]")
from fastapi.testclient import TestClient

import multimind
import multimind.backend.app as backend_app
from multimind.backend.app import SERVICE_MOUNTS, BackendSettings, create_backend_app


@pytest.fixture()
def client():
    return TestClient(create_backend_app(BackendSettings(host="127.0.0.1", port=8080)))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "version": multimind.__version__}


def test_ready(client):
    assert client.get("/ready").status_code == 200


def test_index_accounts_for_every_service(client):
    # Services whose optional extras aren't installed appear under
    # "unavailable" instead of "services"; together they cover all mounts.
    body = client.get("/").json()
    listed = {s["mount"] for s in body["services"]}
    unavailable = {s["mount"] for s in body["unavailable"]}
    assert listed | unavailable == {m for m, _module, _attr, _label in SERVICE_MOUNTS}
    assert not listed & unavailable
    for item in body["unavailable"]:
        assert item["reason"]
    # Dashboard and guard proxy are standalone processes, not mounted here.
    not_mounted_labels = {item["label"] for item in body["not_mounted"]}
    assert "Governance dashboard" in not_mounted_labels
    assert "OpenAI-compatible guard proxy" in not_mounted_labels


def test_each_mounted_service_is_reachable(client):
    services = client.get("/").json()["services"]
    assert services, "no sub-app could be mounted at all"
    for service in services:
        mount_path = service["mount"]
        assert client.get(f"{mount_path}/health").status_code == 200
        assert client.get(f"{mount_path}/docs").status_code == 200
        assert client.get(f"{mount_path}/openapi.json").status_code == 200


def test_missing_extras_degrade_gracefully(monkeypatch):
    # A sub-app whose optional deps are missing must be reported, not fatal.
    broken_module = SERVICE_MOUNTS[0][1]
    real_import = backend_app.import_module

    def fake_import(name, *args, **kwargs):
        if name == broken_module:
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(backend_app, "import_module", fake_import)
    client = TestClient(create_backend_app(BackendSettings()))
    body = client.get("/").json()
    assert SERVICE_MOUNTS[0][0] not in {s["mount"] for s in body["services"]}
    assert any(
        s["mount"] == SERVICE_MOUNTS[0][0] and "simulated missing extra" in s["reason"]
        for s in body["unavailable"]
    )


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("MULTIMIND_BACKEND_HOST", "0.0.0.0")
    monkeypatch.setenv("MULTIMIND_BACKEND_PORT", "9999")
    settings = BackendSettings.from_env()
    assert settings.host == "0.0.0.0"
    assert settings.port == 9999
