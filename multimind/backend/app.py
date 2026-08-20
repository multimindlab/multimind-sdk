"""Single-process aggregator for every MultiMind REST API.

Historically ``gateway/api.py``, ``gateway/rag_api.py``, ``api/unified_api.py``,
and ``api/multi_model_api.py`` were four independent FastAPI apps with no CLI
launcher, each defaulting to (or colliding on) port 8000. :func:`create_backend_app`
mounts all four as sub-applications under one FastAPI instance and one port, so a
frontend only has to know about a single origin.

Each mounted app is a distinct ASGI application, so it keeps its own Swagger UI
and OpenAPI schema at ``<mount>/docs`` / ``<mount>/openapi.json`` (FastAPI does
not merge sub-app schemas into the root's). The root app's own ``/docs`` lists
only the root's landing/health routes; use the per-service links returned by
``GET /`` to reach each service's Swagger.

The governance dashboard (``multimind dashboard``) and the guard proxy
(``multimind serve``) are intentionally NOT mounted here: the dashboard serves
its own static SPA (whose asset paths assume it owns the root), and the proxy
is meant to be OpenAI-SDK-compatible at an unprefixed ``/v1/...``. Both keep
their existing standalone launchers and ports unchanged.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .. import __version__

# (mount path, module path, attribute, human label)
SERVICE_MOUNTS = (
    ("/gateway", "multimind.gateway.api", "app", "API Gateway (chat, sessions, compliance)"),
    ("/rag", "multimind.gateway.rag_api", "app", "RAG API (documents, retrieval, generation)"),
    ("/unified", "multimind.api.unified_api", "app", "Unified multi-modal / MoE API"),
    ("/multimodel", "multimind.api.multi_model_api", "app", "Multi-model wrapper API"),
)


class BackendSettings(BaseModel):
    """Aggregator configuration; build from env with :meth:`from_env`."""

    host: str = "127.0.0.1"
    port: int = 8080

    @classmethod
    def from_env(cls, **overrides: Any) -> "BackendSettings":
        """Read settings from MULTIMIND_BACKEND_* env vars; overrides win."""
        values: Dict[str, Any] = {}
        if raw := os.getenv("MULTIMIND_BACKEND_HOST"):
            values["host"] = raw
        if raw := os.getenv("MULTIMIND_BACKEND_PORT"):
            values["port"] = raw
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


def create_backend_app(settings: Optional[BackendSettings] = None) -> FastAPI:
    """Build the aggregator FastAPI app; mounts every REST API sub-app."""
    settings = settings or BackendSettings.from_env()

    app = FastAPI(
        title="MultiMind Backend",
        description=(
            "Aggregator that mounts every MultiMind REST API under one process "
            "and port. Each service keeps its own Swagger UI at "
            "<mount path>/docs; see GET / for the full list. The governance "
            "dashboard (`multimind dashboard`) and guard proxy (`multimind "
            "serve`) are standalone processes, not mounted here."
        ),
        version=__version__,
    )
    app.state.settings = settings

    # Each sub-app pulls in its own optional extras (e.g. the RAG API needs
    # multimind-sdk[rag]); a partial install should degrade to fewer mounts,
    # not crash the whole backend.
    services = []
    unavailable = []
    for mount_path, module_path, attr, label in SERVICE_MOUNTS:
        try:
            sub_app: FastAPI = getattr(import_module(module_path), attr)
        except ImportError as exc:
            unavailable.append({"label": label, "mount": mount_path, "reason": str(exc)})
            continue
        app.mount(mount_path, sub_app)
        services.append(
            {
                "label": label,
                "mount": mount_path,
                "docs": f"{mount_path}/docs",
                "openapi": f"{mount_path}/openapi.json",
                "health": f"{mount_path}/health",
            }
        )
    app.state.services = services
    app.state.unavailable = unavailable

    @app.get("/health", tags=["system"])
    async def health_check():
        return {"status": "healthy", "version": __version__}

    @app.get("/ready", tags=["system"])
    async def readiness_check():
        return {"status": "ready", "version": __version__}

    @app.get("/", tags=["system"])
    async def index():
        return {
            "name": "MultiMind Backend",
            "version": __version__,
            "services": services,
            "unavailable": unavailable,
            "not_mounted": [
                {
                    "label": "Governance dashboard",
                    "launch": "multimind dashboard",
                    "reason": "serves its own static SPA at /",
                },
                {
                    "label": "OpenAI-compatible guard proxy",
                    "launch": "multimind serve",
                    "reason": "expects unprefixed /v1/... for OpenAI-SDK compatibility",
                },
            ],
        }

    return app
