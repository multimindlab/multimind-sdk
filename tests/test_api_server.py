"""API tests for the MultiMindServer base app."""

import pytest

pytest.importorskip("fastapi", reason="requires multimind-sdk[gateway]")
from fastapi.testclient import TestClient

import multimind
from multimind.server import MultiMindServer


def _client():
    return TestClient(MultiMindServer().get_app())


def test_root():
    r = _client().get("/")
    assert r.status_code == 200
    assert r.json()["version"] == multimind.__version__


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "version": multimind.__version__}


def test_ready():
    r = _client().get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_docs_and_openapi():
    client = _client()
    assert client.get("/docs").status_code == 200
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert r.json()["info"]["version"] == multimind.__version__


def test_add_route():
    server = MultiMindServer()

    async def handler():
        return {"custom": True}

    server.add_route("/custom", handler, methods=["GET"])
    r = TestClient(server.get_app()).get("/custom")
    assert r.status_code == 200
    assert r.json() == {"custom": True}


def test_cors_off_by_default(monkeypatch):
    monkeypatch.delenv("MULTIMIND_CORS_ORIGINS", raising=False)
    server = MultiMindServer()
    middleware_names = [m.cls.__name__ for m in server.get_app().user_middleware]
    assert "CORSMiddleware" not in middleware_names


def test_cors_enabled_via_env(monkeypatch):
    monkeypatch.setenv("MULTIMIND_CORS_ORIGINS", "https://example.com")
    server = MultiMindServer()
    middleware_names = [m.cls.__name__ for m in server.get_app().user_middleware]
    assert "CORSMiddleware" in middleware_names
