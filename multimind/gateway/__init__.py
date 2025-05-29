"""
MultiMind Gateway - Unified API Gateway for Multi-Model Support
"""

__version__ = "0.1.0"

from .api import MultiMindAPI, app
from ..core.models import ModelHandler, ModelResponse
from ..core.config import GatewayConfig

__all__ = [
    "MultiMindAPI",
    "app",
    "ModelHandler",
    "ModelResponse",
    "GatewayConfig"
]