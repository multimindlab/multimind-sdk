"""
Server module for MultiMind SDK.

This module provides server functionality for the MultiMind SDK.
"""

import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__


class MultiMindServer:
    """Basic server for MultiMind SDK."""

    def __init__(self, app_name: str = "MultiMind Server"):
        self.app = FastAPI(
            title=app_name,
            description="Extensible base HTTP server for MultiMind SDK applications.",
            version=__version__,
            openapi_tags=[{"name": "system", "description": "Health and readiness probes"}],
        )
        self.setup_middleware()
        self.setup_routes()

    def setup_middleware(self):
        """Setup CORS middleware (off unless MULTIMIND_CORS_ORIGINS is set)."""
        raw = os.getenv("MULTIMIND_CORS_ORIGINS", "")
        origins = [o.strip() for o in raw.split(",") if o.strip()]
        if origins:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

    def setup_routes(self):
        """Setup basic routes."""

        @self.app.get("/", tags=["system"])
        async def root():
            return {"message": "MultiMind Server is running", "version": __version__}

        @self.app.get("/health", tags=["system"])
        async def health():
            return {"status": "healthy", "version": __version__}

        @self.app.get("/ready", tags=["system"])
        async def ready():
            return {"status": "ready", "version": __version__}

    def add_route(self, path: str, handler, methods: Optional[list] = None):
        """Add a custom route to the server."""
        if methods is None:
            methods = ["GET"]

        for method in methods:
            if method.upper() == "GET":
                self.app.get(path)(handler)
            elif method.upper() == "POST":
                self.app.post(path)(handler)
            elif method.upper() == "PUT":
                self.app.put(path)(handler)
            elif method.upper() == "DELETE":
                self.app.delete(path)(handler)

    async def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the server."""
        import uvicorn

        config = uvicorn.Config(self.app, host=host, port=port)
        server = uvicorn.Server(config)
        await server.serve()

    def get_app(self) -> FastAPI:
        """Get the FastAPI app instance."""
        return self.app


__all__ = ["MultiMindServer"]
