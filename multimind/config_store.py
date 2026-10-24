"""Runtime settings store: a small JSON file edited from the dashboard.

Holds non-secret overrides (default model, log level, per-provider model
names / base URLs / enabled flags, CORS origins, guard-proxy guardrails) that
:class:`multimind.core.config.GatewayConfig` resolves at call time, so a
``PUT /api/config`` takes effect without a restart. API keys are never
accepted: they stay in environment variables.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = "multimind.config.json"
ENV_CONFIG_FILE = "MULTIMIND_CONFIG_FILE"

PROVIDERS = (
    "openai",
    "anthropic",
    "claude",
    "ollama",
    "groq",
    "huggingface",
    "gemini",
    "mistral",
    "deepseek",
    "openrouter",
    "together",
    "xai",
    "perplexity",
    "fireworks",
    "cerebras",
)
PROVIDER_FIELDS = ("model_name", "base_url", "enabled")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
PROXY_STRATEGIES = ("mask", "hash", "remove")
PROXY_FIELDS = ("strategy", "block_on", "budget", "scan_output")
TOP_LEVEL_KEYS = ("default_model", "log_level", "cors_origins")

_SECRET_KEY_RE = re.compile(r"(api_?key|_key$|^key$|token|secret|password|credential)", re.I)


def is_secret_key(key: str) -> bool:
    """True when any dotted segment looks like a credential name."""
    return any(_SECRET_KEY_RE.search(part) for part in key.split("."))


def _pii_types() -> tuple:
    from .compliance.guard import _LABELS

    return tuple(_LABELS)


def _flatten(values: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in values.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict) and not full.startswith("cors_origins"):
            flat.update(_flatten(value, f"{full}."))
        else:
            flat[full] = value
    return flat


def _require_str(key: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _require_bool(key: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value


def _require_str_list(key: str, value: Any) -> list:
    if isinstance(value, str):
        value = [v.strip() for v in value.split(",") if v.strip()]
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return list(dict.fromkeys(v.strip() for v in value))


def validate_setting(key: str, value: Any) -> Any:
    """Return the normalized value for ``key`` or raise ``ValueError``.

    ``None`` is always accepted and means "remove the override".
    """
    if not isinstance(key, str) or not key:
        raise ValueError("setting keys must be non-empty strings")
    if is_secret_key(key):
        raise ValueError(f"{key!r} looks like a credential; API keys stay in environment variables")
    if value is None:
        if key in TOP_LEVEL_KEYS or key.startswith(("providers.", "proxy.")):
            return None
        raise ValueError(f"Unknown setting {key!r}")

    if key == "default_model":
        name = _require_str(key, value).lower()
        if name not in PROVIDERS:
            raise ValueError(f"default_model must be one of {', '.join(PROVIDERS)}")
        return name
    if key == "log_level":
        level = _require_str(key, value).upper()
        if level not in LOG_LEVELS:
            raise ValueError(f"log_level must be one of {', '.join(LOG_LEVELS)}")
        return level
    if key == "cors_origins":
        return _require_str_list(key, value)

    parts = key.split(".")
    if parts[0] == "providers":
        if len(parts) != 3 or parts[1] not in PROVIDERS or parts[2] not in PROVIDER_FIELDS:
            raise ValueError(
                f"Unknown setting {key!r}; use providers.<provider>.<model_name|base_url|enabled>"
            )
        field = parts[2]
        if field == "enabled":
            return _require_bool(key, value)
        return _require_str(key, value)
    if parts[0] == "proxy":
        if len(parts) != 2 or parts[1] not in PROXY_FIELDS:
            raise ValueError(f"Unknown setting {key!r}; use proxy.<{'|'.join(PROXY_FIELDS)}>")
        field = parts[1]
        if field == "strategy":
            strategy = _require_str(key, value)
            if strategy not in PROXY_STRATEGIES:
                raise ValueError(f"proxy.strategy must be one of {', '.join(PROXY_STRATEGIES)}")
            return strategy
        if field == "block_on":
            types = _require_str_list(key, value)
            unknown = sorted(set(types) - set(_pii_types()))
            if unknown:
                raise ValueError(f"Unknown PII types {unknown} in proxy.block_on")
            return types
        if field == "budget":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError("proxy.budget must be a positive number")
            return float(value)
        return _require_bool(key, value)
    raise ValueError(f"Unknown setting {key!r}")


class ConfigStore:
    """Flat, dotted-key JSON store with validated writes."""

    def __init__(self, path: Optional[str] = None):
        self._path = Path(path or os.getenv(ENV_CONFIG_FILE) or DEFAULT_CONFIG_FILE)
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {}
        self.reload()

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.is_file()

    def reload(self) -> Dict[str, Any]:
        """Re-read the file; invalid or secret-looking entries are dropped with a warning."""
        with self._lock:
            loaded: Dict[str, Any] = {}
            if self._path.is_file():
                try:
                    raw = json.loads(self._path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    logger.warning("Could not read %s: %s", self._path, exc)
                    raw = {}
                if not isinstance(raw, dict):
                    logger.warning("%s must contain a JSON object; ignoring", self._path)
                    raw = {}
                for key, value in _flatten(raw).items():
                    try:
                        normalized = validate_setting(key, value)
                    except ValueError as exc:
                        logger.warning("Ignoring %s in %s: %s", key, self._path, exc)
                        continue
                    if normalized is not None:
                        loaded[key] = normalized
            self._data = loaded
            return dict(loaded)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def set_many(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate every entry, then merge and write. Nothing is written on any error."""
        if not isinstance(values, dict):
            raise ValueError("settings payload must be a JSON object")
        normalized = {k: validate_setting(k, v) for k, v in _flatten(values).items()}
        with self._lock:
            for key, value in normalized.items():
                if value is None:
                    self._data.pop(key, None)
                else:
                    self._data[key] = value
            self._write()
            return dict(self._data)

    def _write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self._path)


_store: Optional[ConfigStore] = None
_store_lock = threading.Lock()


def get_config_store() -> ConfigStore:
    """Process-wide store; the path is fixed on first use (``MULTIMIND_CONFIG_FILE``)."""
    global _store
    with _store_lock:
        if _store is None:
            _store = ConfigStore()
        return _store


def reset_config_store(path: Optional[str] = None) -> ConfigStore:
    """Replace the singleton (tests, or after changing ``MULTIMIND_CONFIG_FILE``)."""
    global _store
    with _store_lock:
        _store = ConfigStore(path)
        return _store
