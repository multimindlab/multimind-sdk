"""
MultiMind Gateway Package.
Provides a unified interface for all MultiMind services.
"""

__version__ = "1.0.0"

# API classes
from .api import MultiMindAPI, app, start
from .compliance_api import router as compliance_router

# Model handlers
from .models import OpenAIHandler, AnthropicHandler, OllamaHandler, HuggingFaceHandler

__all__ = [
    # API
    "MultiMindAPI",
    "app",
    "start",
    "compliance_router",
    
    # Model handlers
    "OpenAIHandler",
    "AnthropicHandler",
    "OllamaHandler",
    "HuggingFaceHandler",
]