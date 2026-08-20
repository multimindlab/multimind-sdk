"""Aggregator that serves every MultiMind REST API from one process/port.

See :mod:`multimind.backend.app` for :func:`create_backend_app`.
"""

from .app import SERVICE_MOUNTS, BackendSettings, create_backend_app

__all__ = ["BackendSettings", "SERVICE_MOUNTS", "create_backend_app"]
