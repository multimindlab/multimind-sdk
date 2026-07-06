"""API tests for multimind.api.unified_api using TestClient with a mocked router."""

import pytest
from fastapi.testclient import TestClient

import multimind
import multimind.api.unified_api as ua


class FakeRouter:
    def __init__(self):
        self.modality_registry = {"text": {"fake-model": object()}}
        self.cost_tracker = type("CT", (), {"costs": {}})()
        self.performance_metrics = type("PM", (), {"metrics": {}})()

    async def route_request(self, request):
        return {"text": "routed response"}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setattr(ua, "_get_router", lambda: FakeRouter())
    return TestClient(ua.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "version": multimind.__version__}


def test_ready(client):
    assert client.get("/ready").status_code == 200


def test_docs_and_openapi(client):
    assert client.get("/docs").status_code == 200
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert schema["info"]["version"] == multimind.__version__
    assert "/v1/process" in schema["paths"]


def test_process_router_happy_path(client):
    r = client.post(
        "/v1/process",
        json={"inputs": [{"content": "hello", "modality": "text"}], "use_moe": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["outputs"] == {"text": "routed response"}
    assert body["metrics"]["processing_type"] == "router"


def test_process_validation_error(client):
    r = client.post("/v1/process", json={"inputs": "not-a-list"})
    assert r.status_code == 422


def test_process_moe_no_experts_returns_400(client, monkeypatch):
    monkeypatch.setattr(ua, "_build_experts", lambda modalities, router: {})
    r = client.post(
        "/v1/process",
        json={"inputs": [{"content": "hello", "modality": "text"}], "use_moe": True},
    )
    assert r.status_code == 400


def test_list_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json() == {"models": {"text": ["fake-model"]}}


def test_list_workflows(client):
    r = client.get("/v1/workflows")
    assert r.status_code == 200
    assert "workflows" in r.json()


def test_metrics(client):
    r = client.get("/v1/metrics")
    assert r.status_code == 200
    assert set(r.json().keys()) == {"costs", "performance"}
