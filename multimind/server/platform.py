"""One FastAPI application and one OpenAPI document for the supported HTTP API.

Routes are copied from the gateway (chat, sessions, compliance), the governance
dashboard (audit, costs, inventory, guardrails, runtime config), and the guard
proxy (OpenAI-compatible chat and embeddings, guardrail reload). The dashboard
UI is served at ``/`` from the same process, so the page calls ``/v1/*`` and
``/api/*`` on its own origin. The multi-modal MoE app is not included. RAG
routes are included when the RAG extra imports.

Authentication: when ``API_KEYS`` is set (comma-separated), every route except
health, readiness, and the docs pages requires an ``X-API-Key`` header. When
``API_KEYS`` is unset, the API is open, matching local development.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Set

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config_store import get_config_store

_DOC_PATHS = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
_OPEN_PATHS = {"/health", "/ready", "/", *_DOC_PATHS}
_STATIC_DIR = Path(__file__).parent.parent / "dashboard" / "static"


def _iter_routes(router) -> Iterable[object]:
    """Yield concrete routes, descending into routers FastAPI >=0.130 includes lazily."""
    for route in router.routes:
        inner = getattr(route, "original_router", None)
        if inner is not None:
            yield from _iter_routes(inner)
        else:
            yield route


def _copy_routes(
    source: FastAPI, dest: FastAPI, paths: Optional[Iterable[str]] = None
) -> List[str]:
    """Attach route objects from ``source`` onto ``dest``. Returns copied paths."""
    allow = set(paths) if paths is not None else None
    already = {getattr(route, "path", None) for route in _iter_routes(dest.router)}
    copied: List[str] = []
    for route in _iter_routes(source.router):
        path = getattr(route, "path", None)
        if not path or path in _DOC_PATHS or path in already:
            continue
        if allow is not None and path not in allow:
            continue
        dest.router.routes.append(route)
        copied.append(path)
    return copied


def create_platform_app() -> FastAPI:
    """Build the supported API. Swagger UI is at ``/docs``."""
    app = FastAPI(
        title="MultiMind API",
        description=(
            "Supported MultiMind HTTP API: chat and generation, sessions, "
            "compliance checks that do not invent a pass, governance artifacts "
            "(audit, costs, inventory, guardrails), and the OpenAI-compatible "
            "guard proxy. Set API_KEYS to require an X-API-Key header."
        ),
        version=__version__,
        openapi_tags=[
            {"name": "models", "description": "Provider availability"},
            {"name": "generation", "description": "Chat, generation, and comparison"},
            {"name": "sessions", "description": "Chat sessions"},
            {"name": "compliance", "description": "Implemented compliance checks"},
            {"name": "governance", "description": "Audit, costs, inventory, guardrails"},
            {"name": "proxy", "description": "OpenAI-compatible guarded proxy"},
            {"name": "system", "description": "Health and readiness"},
        ],
    )

    # Dashboard UI files are open so the page can load and then send X-API-Key itself.
    ui_paths: Set[str] = {"/config.js", "/index.html"}
    if _STATIC_DIR.is_dir():
        ui_paths.update(f"/{p.name}" for p in _STATIC_DIR.iterdir() if p.is_file())

    @app.middleware("http")
    async def require_api_key(request, call_next):
        path = request.url.path
        if path in _OPEN_PATHS or path in ui_paths or path.startswith("/docs"):
            return await call_next(request)
        raw = os.getenv("API_KEYS", "")
        keys = [item.strip() for item in raw.split(",") if item.strip()]
        if not keys:
            return await call_next(request)
        if request.headers.get("x-api-key") not in keys:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
        return await call_next(request)

    # Added after auth so browser preflight is answered before the key check.
    # Config store wins over env; read once here, so a change needs a restart.
    origins = get_config_store().get("cors_origins") or [
        item.strip()
        for item in (
            os.getenv("MULTIMIND_CORS_ORIGINS") or os.getenv("MULTIMIND_ALLOWED_ORIGINS") or ""
        ).split(",")
        if item.strip()
    ]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    from ..gateway.api import app as gateway_app

    _copy_routes(gateway_app, app)

    try:
        from ..dashboard.server import DashboardSettings, create_dashboard_app

        dashboard = create_dashboard_app(DashboardSettings.from_env())
        api_paths = [
            path
            for path in (getattr(route, "path", "") for route in dashboard.router.routes)
            if path.startswith("/api")
        ]
        _copy_routes(dashboard, app, paths=api_paths)

        # Never embed a key here: this file is open and would hand out the API key.
        # The page is same-origin, so no base URL is needed; the key comes from the UI.
        @app.get("/config.js", include_in_schema=False)
        async def platform_config_js():
            return Response(
                content='window.MULTIMIND_API_BASE="";\nwindow.MULTIMIND_API_KEY="";\n',
                media_type="application/javascript",
                headers={"Cache-Control": "no-store"},
            )

        serve_ui = _STATIC_DIR.is_dir()
    except ImportError:
        serve_ui = False

    try:
        from ..gateway.guard_proxy import create_app as create_proxy

        proxy = create_proxy()
        _copy_routes(
            proxy,
            app,
            paths={"/v1/chat/completions", "/v1/embeddings", "/v1/guardrails/reload"},
        )
    except ImportError:
        pass

    try:
        from ..gateway.rag_api import app as rag_app

        _copy_routes(rag_app, app)
        app.state.rag = "mounted"
    except ImportError:
        app.state.rag = "unavailable"

    if serve_ui:
        # The gateway's JSON "/" gives way to the dashboard page; static files are matched last.
        app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != "/"]
        app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="dashboard-ui")
        app.state.dashboard_ui = "mounted"
    else:
        app.state.dashboard_ui = "unavailable"

    return app
