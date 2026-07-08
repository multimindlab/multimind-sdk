"""
Client module for MultiMind SDK.

This module provides client interfaces for connecting to various services.
ModelClient and FederatedRouter need torch, so they resolve lazily.
"""

from __future__ import annotations

from typing import Any

from multimind._lazy import lazy_attr

from .model_session import ModelSession
from .rag_client import RAGClient

_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    "FederatedRouter": ("multimind.client.federated_router", "finetune"),
    "ModelClient": ("multimind.client.model_client", "finetune"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_path, extras_group = _LAZY_ATTRS[name]
        value = lazy_attr(name, module_path, extras_group)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'multimind.client' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))


__all__ = ["FederatedRouter", "ModelClient", "ModelSession", "RAGClient"]
