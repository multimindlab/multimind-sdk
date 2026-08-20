"""Tests for multimind.backend (the single-process REST API aggregator)."""

import pytest

pytest.importorskip("fastapi", reason="requires multimind-sdk[gateway]")
from fastapi.testclient import TestClient

import multimind
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


def test_index_lists_every_mounted_service(client):
    body = client.get("/").json()
    listed_mounts = {s["mount"] for s in body["services"]}
    assert listed_mounts == {mount for mount, _module, _attr, _label in SERVICE_MOUNTS}
    # Dashboard and guard proxy are standalone processes, not mounted here.
    not_mounted_labels = {item["label"] for item in body["not_mounted"]}
    assert "Governance dashboard" in not_mounted_labels
    assert "OpenAI-compatible guard proxy" in not_mounted_labels


@pytest.mark.parametrize("mount_path", [m[0] for m in SERVICE_MOUNTS])
def test_each_mounted_service_is_reachable(client, mount_path):
    assert client.get(f"{mount_path}/health").status_code == 200
    assert client.get(f"{mount_path}/docs").status_code == 200
    assert client.get(f"{mount_path}/openapi.json").status_code == 200


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("MULTIMIND_BACKEND_HOST", "0.0.0.0")
    monkeypatch.setenv("MULTIMIND_BACKEND_PORT", "9999")
    settings = BackendSettings.from_env()
    assert settings.host == "0.0.0.0"
    assert settings.port == 9999
