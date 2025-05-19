"""
MultiMind Gateway - Unified CLI and API Gateway for Multi-Model Support
"""

__version__ = "0.1.0"

from .cli import MultiMindCLI
from .api import MultiMindAPI
from .models import ModelHandler, ModelResponse
from .config import GatewayConfig

__all__ = [
    "MultiMindCLI",
    "MultiMindAPI",
    "ModelHandler",
    "ModelResponse",
    "GatewayConfig"
] 