"""
MultiMind Gateway Package.
Provides a unified interface for all MultiMind services.
"""

__version__ = "1.0.0"

from .api import app, start
from .compliance_api import router as compliance_router

__all__ = [
    "app",
    "start",
    "compliance_router"
]