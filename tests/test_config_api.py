"""Dashboard /api/config routes: effective config, partial updates, provider test.

No live provider calls; get_model_handler is mocked for the test-provider route.
"""

import json
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("fastapi", reason="requires multimind-sdk[gateway]")
from fastapi.testclient import TestClient

from multimind.config_store import reset_config_store
from multimind.dashboard.server import DashboardSettings, create_dashboard_app

SECRET = "sk-test-secret-value-1234567890"


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTIMIND_CONFIG_FILE", str(tmp_path / "multimind.config.json"))
    reset_config_store()
    yield
    monkeypatch.delenv("MULTIMIND_CONFIG_FILE", raising=False)
    reset_config_store()


@pytest.fixture()
def settings(tmp_path):
    return DashboardSettings(
        audit_log=str(tmp_path / "audit.jsonl"),
        costs_log=str(tmp_path / "costs.jsonl"),
        project_path=str(tmp_path),
        guardrails_path=str(tmp_path / "guardrails.json"),
    )


@pytest.fixture()
def client(settings):
    return TestClient(create_dashboard_app(settings))


def test_get_config_shape_and_no_secrets(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    response = client.get("/api/config")
    assert response.status_code == 200
    body = response.json()
    assert SECRET not in response.text
    assert "api_key" not in body["effective"]["providers"]["openai"]
    assert body["exists"] is False
    assert body["overrides"] == {}
    assert set(body["effective"]) == {
        "default_model",
        "log_level",
        "cors_origins",
        "proxy",
        "providers",
    }
    openai = body["effective"]["providers"]["openai"]
    assert set(openai) == {"model_name", "base_url", "enabled", "has_api_key"}
    assert openai["enabled"] is True
    assert body["effective"]["providers"]["ollama"]["base_url"] == "http://localhost:11434"
    assert body["effective"]["proxy"] == {
        "strategy": "mask",
        "block_on": [],
        "budget": None,
        "scan_output": True,
    }


def test_has_api_key_reflects_env_without_exposing_it(client, monkeypatch):
    # GatewayConfig captures keys at import, so this checks the flag is a bool and never the value.
    monkeypatch.setenv("XAI_API_KEY", SECRET)
    body = client.get("/api/config").json()
    assert isinstance(body["effective"]["providers"]["xai"]["has_api_key"], bool)
    assert SECRET not in json.dumps(body)


def test_put_config_changes_effective_values_without_restart(client, tmp_path):
    before = client.get("/api/config").json()
    assert before["effective"]["default_model"] != "groq"

    response = client.put(
        "/api/config",
        json={
            "default_model": "groq",
            "log_level": "debug",
            "providers": {"openai": {"model_name": "gpt-4o-mini", "enabled": False}},
            "providers.ollama.base_url": "http://gpu-box:11434",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["exists"] is True
    assert body["effective"]["default_model"] == "groq"
    assert body["effective"]["log_level"] == "DEBUG"
    assert body["effective"]["providers"]["openai"]["model_name"] == "gpt-4o-mini"
    assert body["effective"]["providers"]["openai"]["enabled"] is False
    assert body["effective"]["providers"]["ollama"]["base_url"] == "http://gpu-box:11434"
    assert "apply_hint" in body

    after = client.get("/api/config").json()
    assert after["effective"]["default_model"] == "groq"
    assert after["overrides"]["providers.openai.enabled"] is False
    assert json.loads((tmp_path / "multimind.config.json").read_text())["default_model"] == "groq"

    # partial update leaves other keys alone; null clears one
    client.put("/api/config", json={"providers.openai.enabled": None})
    after = client.get("/api/config").json()
    assert after["effective"]["providers"]["openai"]["enabled"] is True
    assert after["effective"]["default_model"] == "groq"


@pytest.mark.parametrize(
    "payload",
    [
        {"default_model": "nope"},
        {"providers.openai.model_name": ""},
        {"proxy.budget": -5},
        {"something_else": True},
        {"openai_api_key": "sk-x"},
        {"providers": {"openai": {"api_key": "sk-x"}}},
    ],
)
def test_put_config_rejects_invalid_and_secret_keys(client, payload, tmp_path):
    response = client.put("/api/config", json=payload)
    assert response.status_code == 422, response.text
    assert not (tmp_path / "multimind.config.json").exists()
    assert "sk-x" not in json.dumps(client.get("/api/config").json())


def test_put_config_proxy_keys_sync_guardrails_file(client, settings):
    response = client.put(
        "/api/config",
        json={"proxy.strategy": "hash", "proxy.block_on": ["ssn"], "proxy.budget": 3},
    )
    assert response.status_code == 200
    assert response.json()["effective"]["proxy"] == {
        "strategy": "hash",
        "block_on": ["ssn"],
        "budget": 3.0,
        "scan_output": True,
    }
    on_disk = json.loads(open(settings.guardrails_path).read())
    assert on_disk == {
        "strategy": "hash",
        "block_on": ["ssn"],
        "budget_max_cost": 3.0,
        "scan_output": True,
    }
    assert client.get("/api/guardrails").json()["config"]["strategy"] == "hash"


def test_put_config_requires_dashboard_key_when_set(tmp_path):
    settings = DashboardSettings(
        audit_log=str(tmp_path / "a.jsonl"),
        costs_log=str(tmp_path / "c.jsonl"),
        project_path=str(tmp_path),
        guardrails_path=str(tmp_path / "guardrails.json"),
        write_api_key="secret123",
    )
    client = TestClient(create_dashboard_app(settings))
    assert client.get("/api/config").status_code == 200
    assert client.put("/api/config", json={"default_model": "groq"}).status_code == 401
    wrong = client.put("/api/config", json={"default_model": "groq"}, headers={"X-API-Key": "no"})
    assert wrong.status_code == 401
    ok = client.put(
        "/api/config", json={"default_model": "groq"}, headers={"X-API-Key": "secret123"}
    )
    assert ok.status_code == 200
    assert client.get("/api/config").json()["effective"]["default_model"] == "groq"


def test_test_provider_success(client, monkeypatch):
    from multimind.gateway import models as gateway_models

    handler = AsyncMock()
    handler.chat = AsyncMock(return_value={"content": "pong"})
    calls = []

    def fake_handler(name):
        calls.append(name)
        return handler

    monkeypatch.setattr(gateway_models, "get_model_handler", fake_handler)
    response = client.post("/api/config/test-provider", json={"provider": "OpenAI"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True and body["error"] is None
    assert isinstance(body["latency_ms"], int)
    assert calls == ["openai"]
    handler.chat.assert_awaited_once()
    messages = handler.chat.await_args.args[0]
    assert messages == [{"role": "user", "content": "ping"}]


def test_test_provider_error_is_reported_and_scrubbed(client, monkeypatch):
    from multimind.gateway import models as gateway_models

    monkeypatch.setenv("OPENAI_API_KEY", SECRET)

    def boom(name):
        raise RuntimeError(f"401 invalid key {SECRET}")

    monkeypatch.setattr(gateway_models, "get_model_handler", boom)
    response = client.post("/api/config/test-provider", json={"provider": "openai"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert "401 invalid key" in body["error"]
    assert SECRET not in response.text


def test_test_provider_unknown_provider(client):
    response = client.post("/api/config/test-provider", json={"provider": "nope"})
    assert response.status_code == 422
