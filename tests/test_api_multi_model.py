"""API tests for multimind.api.multi_model_api using TestClient with mocked models."""

import pytest

pytest.importorskip("fastapi", reason="requires multimind-sdk[gateway]")
from fastapi.testclient import TestClient

import multimind
import multimind.api.multi_model_api as mma


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    return TestClient(mma.app)


class FakeWrapper:
    def __init__(self):
        self.models = {"fake": object()}

    async def generate(self, prompt, **kwargs):
        return f"echo: {prompt}"

    async def chat(self, messages, **kwargs):
        return "chat reply"

    async def embeddings(self, text, **kwargs):
        return [[0.1, 0.2, 0.3]]


@pytest.fixture()
def mocked_wrapper(monkeypatch):
    wrapper = FakeWrapper()

    async def fake_get_multi_model(**kwargs):
        return wrapper

    monkeypatch.setattr(mma, "_get_multi_model", fake_get_multi_model)
    return wrapper


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "version": multimind.__version__}


def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_docs_and_openapi(client):
    assert client.get("/docs").status_code == 200
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert schema["info"]["version"] == multimind.__version__
    assert "/generate" in schema["paths"]


def test_generate_happy_path(client, mocked_wrapper):
    r = client.post("/generate", json={"prompt": "hello"})
    assert r.status_code == 200
    assert r.json() == {"response": "echo: hello"}


def test_generate_validation_error(client):
    r = client.post("/generate", json={})
    assert r.status_code == 422


def test_generate_no_models_returns_503(client, monkeypatch):
    wrapper = FakeWrapper()
    wrapper.models = {}

    async def fake_get_multi_model(**kwargs):
        return wrapper

    monkeypatch.setattr(mma, "_get_multi_model", fake_get_multi_model)
    r = client.post("/generate", json={"prompt": "hello"})
    assert r.status_code == 503


def test_chat_happy_path(client, mocked_wrapper):
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json() == {"response": "chat reply"}


def test_chat_validation_error(client):
    r = client.post("/chat", json={"messages": "not-a-list"})
    assert r.status_code == 422


def test_embeddings_happy_path(client, mocked_wrapper):
    r = client.post("/embeddings", json={"text": "hi"})
    assert r.status_code == 200
    assert r.json() == {"embeddings": [[0.1, 0.2, 0.3]]}


def test_embeddings_validation_error(client):
    r = client.post("/embeddings", json={})
    assert r.status_code == 422


def test_api_key_enforced_when_configured(monkeypatch, mocked_wrapper):
    monkeypatch.setenv("API_KEYS", "secret-key")
    client = TestClient(mma.app)
    r = client.post("/generate", json={"prompt": "hello"})
    assert r.status_code == 401
    r = client.post("/generate", json={"prompt": "hello"}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401
    r = client.post("/generate", json={"prompt": "hello"}, headers={"X-API-Key": "secret-key"})
    assert r.status_code == 200
    # health stays open
    assert client.get("/health").status_code == 200
