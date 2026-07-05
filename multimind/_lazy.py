"""Lazy import helpers for MultiMind SDK.

Heavy optional dependencies (torch, chromadb, fastapi, faiss, …) are kept out of
``multimind/__init__.py`` so that ``pip install multimind-sdk`` followed by
``from multimind import OpenAIModel`` works without dragging in unrelated extras.

When a user actually touches one of those features (``from multimind import RAG``,
``from multimind import LoRATrainer``, …) we import the relevant subpackage on
demand and re-raise with a friendly error pointing at the right extras group.

This module is intentionally tiny — it only depends on the Python standard
library and is safe to import at the very top of ``multimind/__init__.py``.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


def import_optional(
    module_path: str,
    extras_group: str,
    *,
    package_name: str = "multimind-sdk",
) -> ModuleType:
    """Import ``module_path``; raise a helpful ``ImportError`` on failure.

    Parameters
    ----------
    module_path:
        Dotted module path to import, e.g. ``"multimind.rag"``.
    extras_group:
        Name of the extras group that installs the missing dependency,
        e.g. ``"rag"`` or ``"finetune"``. Surfaced in the error message.
    package_name:
        Distribution name shown in the install hint. Defaults to
        ``"multimind-sdk"``.
    """
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"`{module_path}` requires additional dependencies. "
            f"Install with: pip install '{package_name}[{extras_group}]'"
        ) from exc


# Substring that identifies an ImportError already re-raised by a MultiMind
# subpackage with a friendly install hint. Kept loose so it matches both
# "X requires additional dependencies" and "X features require additional
# dependencies" phrasings.
_FRIENDLY_MARKER = "additional dependencies. Install with: pip install"


def lazy_attr(
    name: str,
    module_path: str,
    extras_group: str | None = None,
    *,
    package_name: str = "multimind-sdk",
) -> Any:
    """Resolve attribute ``name`` from ``module_path`` lazily.

    Used by package-level ``__getattr__`` (PEP 562). If the target module fails
    to import and ``extras_group`` is provided, the raised ``ImportError`` tells
    the caller how to install the missing extras. If ``extras_group`` is None,
    the original ImportError propagates unchanged.

    When the underlying ImportError was *already* re-raised with a friendly
    install hint by a subpackage's own ``__init__.py``, we pass it through
    instead of double-wrapping with a less specific message.
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        if extras_group is None:
            raise
        if _FRIENDLY_MARKER in str(exc):
            # The subpackage already gave a friendly, often more specific
            # message (e.g. `finetune` vs `finetune-gpu`). Don't clobber it.
            raise
        raise ImportError(
            f"`{name}` requires additional dependencies. "
            f"Install with: pip install '{package_name}[{extras_group}]'"
        ) from exc

    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise ImportError(
            f"Could not resolve `{name}` from `{module_path}`. "
            "This is a MultiMind SDK packaging bug; please report it."
        ) from exc


__all__ = ["import_optional", "lazy_attr"]
