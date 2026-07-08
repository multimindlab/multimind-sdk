"""
API module for MultiMind SDK.

This module provides FastAPI-based API interfaces for the MultiMind SDK.
The FastAPI apps need the ``gateway`` extras, so they resolve lazily —
importing this package (e.g. transitively via ``multimind.router``) does
not require fastapi to be installed.
"""

from __future__ import annotations

from typing import Any

from multimind._lazy import lazy_attr

# Public name -> (dotted module path, attribute name on that module).
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "multi_model_app": ("multimind.api.multi_model_api", "app"),
    "unified_app": ("multimind.api.unified_api", "app"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_path, attr = _LAZY_ATTRS[name]
        value = lazy_attr(attr, module_path, "gateway")
        globals()[name] = value
        return value
    raise AttributeError(f"module 'multimind.api' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))


__all__ = ["multi_model_app", "unified_app"]
