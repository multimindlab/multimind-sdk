"""API tests for multimind.gateway.api using TestClient with mocked model handlers."""

import pytest

pytest.importorskip("fastapi", reason="requires multimind-sdk[gateway]")
from fastapi.testclient import TestClient

import multimind
import multimind.gateway.api as gw
from multimind.core.models import ModelResponse


class FakeHandler:
    async def chat(self, messages, **kwargs):
        return ModelResponse(content="fake chat", model="fake", usage={"total_tokens": 3})

    async def generate(self, prompt, **kwargs):
        return ModelResponse(content="fake generation", model="fake")


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(
        gw,
        "_model_status",
        lambda: {"openai": True, "anthropic": False, "ollama": False},
    )
    monkeypatch.setattr(gw, "get_model_handler", lambda model: FakeHandler())
    return TestClient(gw.app)


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["version"] == multimind.__version__


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
    assert "/v1/chat" in schema["paths"]


def test_list_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["models"]["openai"]["status"] == "available"


def test_list_models_none_configured_returns_503(client, monkeypatch):
    monkeypatch.setattr(gw, "_model_status", lambda: {"openai": False})
    r = client.get("/v1/models")
    assert r.status_code == 503


def test_chat_happy_path(client):
    r = client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "model": "openai"},
    )
    assert r.status_code == 200
    assert r.json()["content"] == "fake chat"


def test_chat_validation_error(client):
    r = client.post("/v1/chat", json={})
    assert r.status_code == 422


def test_chat_unavailable_model_returns_400(client):
    r = client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "model": "anthropic"},
    )
    assert r.status_code == 400


def test_generate_happy_path(client):
    r = client.post("/v1/generate", json={"prompt": "hello", "model": "openai"})
    assert r.status_code == 200
    assert r.json()["content"] == "fake generation"


def test_generate_validation_error(client):
    r = client.post("/v1/generate", json={"model": "openai"})
    assert r.status_code == 422


def test_compare(client):
    r = client.post("/v1/compare", json={"prompt": "hello", "models": ["openai", "anthropic"]})
    assert r.status_code == 200
    # anthropic is unavailable and skipped; openai responds
    assert list(r.json()["responses"].keys()) == ["openai"]


def test_metrics(client):
    r = client.get("/v1/metrics")
    assert r.status_code == 200
    assert "metrics" in r.json()


def test_session_lifecycle(client):
    r = client.post("/v1/sessions", json={"model": "openai"})
    assert r.status_code == 200
    session_id = r.json()["session_id"]

    r = client.get("/v1/sessions")
    assert r.status_code == 200
    assert any(s["session_id"] == session_id for s in r.json())

    r = client.get(f"/v1/sessions/{session_id}")
    assert r.status_code == 200

    r = client.post(
        f"/v1/sessions/{session_id}/messages",
        json={"role": "user", "content": "hello"},
    )
    assert r.status_code == 200

    r = client.delete(f"/v1/sessions/{session_id}")
    assert r.status_code == 200

    assert client.get(f"/v1/sessions/{session_id}").status_code == 404


def test_session_validation_error(client):
    r = client.post("/v1/sessions", json={})
    assert r.status_code == 422


def test_compliance_regulations(client):
    r = client.get("/v1/compliance/regulations")
    assert r.status_code == 200
    assert "GDPR" in r.json()


def test_compliance_monitor_unknown_regulation_returns_400(client):
    r = client.post(
        "/v1/compliance/monitor",
        json={
            "organization_id": "org",
            "organization_name": "Org",
            "dpo_email": "dpo@example.com",
            "enabled_regulations": ["NOT_A_REGULATION"],
            "compliance_rules": {},
        },
    )
    assert r.status_code == 400


def test_compliance_monitor_validation_error(client):
    r = client.post("/v1/compliance/monitor", json={"organization_id": "org"})
    assert r.status_code == 422
