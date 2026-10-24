"""Unit tests for the runtime config store (multimind.config_store)."""

import json

import pytest

from multimind.config_store import (
    PROVIDERS,
    ConfigStore,
    get_config_store,
    reset_config_store,
    validate_setting,
)


@pytest.fixture()
def store(tmp_path):
    return ConfigStore(str(tmp_path / "multimind.config.json"))


def test_empty_store_without_file(store):
    assert store.as_dict() == {}
    assert store.exists() is False
    assert store.get("default_model", "openai") == "openai"


def test_set_many_writes_and_reloads(store):
    out = store.set_many({"default_model": "Groq", "providers.openai.model_name": "gpt-4o-mini"})
    assert out == {"default_model": "groq", "providers.openai.model_name": "gpt-4o-mini"}
    assert store.exists()
    on_disk = json.loads(store.path.read_text())
    assert on_disk["default_model"] == "groq"

    fresh = ConfigStore(str(store.path))
    assert fresh.get("providers.openai.model_name") == "gpt-4o-mini"


def test_nested_payload_is_flattened(store):
    store.set_many({"providers": {"ollama": {"base_url": "http://box:11434", "enabled": False}}})
    assert store.get("providers.ollama.base_url") == "http://box:11434"
    assert store.get("providers.ollama.enabled") is False


def test_none_clears_an_override(store):
    store.set_many({"log_level": "debug"})
    assert store.get("log_level") == "DEBUG"
    store.set_many({"log_level": None})
    assert "log_level" not in store.as_dict()


@pytest.mark.parametrize(
    "payload",
    [
        {"unknown_key": 1},
        {"default_model": "nope"},
        {"default_model": ""},
        {"log_level": "LOUD"},
        {"providers.notaprovider.model_name": "x"},
        {"providers.openai.model_name": ""},
        {"providers.openai.model_name": 3},
        {"providers.openai.enabled": "yes"},
        {"providers.openai.temperature": 0.1},
        {"cors_origins": "not-a-list-of-strings-or-is-it"},  # string form is accepted; see below
        {"proxy.strategy": "shred"},
        {"proxy.block_on": ["not_a_pii_type"]},
        {"proxy.budget": 0},
        {"proxy.budget": -1},
        {"proxy.budget": True},
        {"proxy.scan_output": "false"},
        {"proxy.timeout": 5},
    ],
)
def test_invalid_values_are_rejected_and_nothing_written(store, payload):
    if payload == {"cors_origins": "not-a-list-of-strings-or-is-it"}:
        # comma strings are accepted and split
        assert store.set_many(payload)["cors_origins"] == ["not-a-list-of-strings-or-is-it"]
        return
    with pytest.raises(ValueError):
        store.set_many(payload)
    assert not store.exists()


def test_all_or_nothing_write(store):
    with pytest.raises(ValueError):
        store.set_many({"default_model": "groq", "log_level": "nope"})
    assert store.as_dict() == {}


@pytest.mark.parametrize(
    "key",
    [
        "openai_api_key",
        "providers.openai.api_key",
        "providers.openai.apiKey",
        "api_key",
        "token",
        "proxy.secret",
        "providers.groq.password",
        "hf_token",
    ],
)
def test_secret_like_keys_are_refused(store, key):
    with pytest.raises(ValueError, match="credential"):
        store.set_many({key: "sk-should-never-be-stored"})
    assert not store.exists()


def test_proxy_settings_validate_like_guardrails(store):
    out = store.set_many(
        {
            "proxy.strategy": "hash",
            "proxy.block_on": "ssn,email,ssn",
            "proxy.budget": 12,
            "proxy.scan_output": False,
        }
    )
    assert out["proxy.block_on"] == ["ssn", "email"]
    assert out["proxy.budget"] == 12.0 and isinstance(out["proxy.budget"], float)
    assert out["proxy.scan_output"] is False


def test_reload_drops_invalid_and_secret_entries(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(
        json.dumps(
            {
                "default_model": "mistral",
                "openai_api_key": "sk-leak",
                "providers": {"openai": {"model_name": "gpt-4o", "enabled": "yes"}},
                "bogus": 1,
            }
        )
    )
    store = ConfigStore(str(path))
    assert store.as_dict() == {"default_model": "mistral", "providers.openai.model_name": "gpt-4o"}

    path.write_text(json.dumps({"default_model": "xai"}))
    assert store.get("default_model") == "mistral"
    store.reload()
    assert store.get("default_model") == "xai"


def test_corrupt_file_is_ignored(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text("{not json")
    assert ConfigStore(str(path)).as_dict() == {}
    path.write_text("[1, 2]")
    assert ConfigStore(str(path)).as_dict() == {}


def test_singleton_follows_env_path(tmp_path, monkeypatch):
    target = tmp_path / "from-env.json"
    monkeypatch.setenv("MULTIMIND_CONFIG_FILE", str(target))
    store = reset_config_store()
    assert store.path == target
    assert get_config_store() is store
    store.set_many({"default_model": "cerebras"})
    assert json.loads(target.read_text())["default_model"] == "cerebras"
    monkeypatch.delenv("MULTIMIND_CONFIG_FILE")
    reset_config_store()


def test_validate_setting_covers_every_provider():
    for name in PROVIDERS:
        assert validate_setting(f"providers.{name}.enabled", True) is True
        assert validate_setting(f"providers.{name}.model_name", " m ") == "m"
