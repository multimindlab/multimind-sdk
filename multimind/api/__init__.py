"""
API module for MultiMind SDK.

This module provides FastAPI-based API interfaces for the MultiMind SDK.
"""

from .multi_model_api import app as multi_model_app
from .unified_api import app as unified_app

__all__ = [
    "multi_model_app",
    "unified_app"
] 