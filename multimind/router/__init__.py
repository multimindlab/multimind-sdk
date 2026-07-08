"""
Router module for MultiMind SDK.

This module provides routing capabilities for directing requests to appropriate models
and handling fallback strategies.

``AdaptiveRouter`` needs numpy (via its ``ImportanceScorer`` dependency), so it
resolves lazily; everything else here is stdlib-only and stays eager.
"""

from __future__ import annotations

from typing import Any

from .._lazy import lazy_attr
from .fallback import FallbackHandler
from .multi_modal_router import MultiModalRouter
from .router import ModelRouter
from .strategy import CostAwareStrategy, HybridStrategy, LatencyAwareStrategy, RoutingStrategy

# Import Router from core (fix the circular import issue)
try:
    from ..core.router import Router, TaskConfig, TaskType
except ImportError:
    # Fallback if core router not available
    Router = ModelRouter
    TaskType = None
    TaskConfig = None

_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    "AdaptiveRouter": ("multimind.router.adaptive", "rag"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_path, extras_group = _LAZY_ATTRS[name]
        value = lazy_attr(name, module_path, extras_group)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'multimind.router' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))


__all__ = [
    "AdaptiveRouter",
    "FallbackHandler",
    "MultiModalRouter",
    "ModelRouter",
    "Router",
    "TaskType",
    "TaskConfig",
    "RoutingStrategy",
    "CostAwareStrategy",
    "LatencyAwareStrategy",
    "HybridStrategy",
]
