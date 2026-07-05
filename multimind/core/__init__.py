"""
MultiMind Core - Shared functionality for the MultiMind project
"""

__version__ = "0.1.0"

from .base import BaseLLM
from .chat import ChatManager, ChatMessage, ChatSession, chat_manager
from .config import GatewayConfig, ModelConfig, config
from .exceptions import ConfigurationError
from .local_runner import LocalRunner
from .models import ModelHandler, ModelResponse
from .monitoring import ModelHealth, ModelMetrics, ModelMonitor, monitor
from .multimind import MultiMind
from .provider import ProviderAdapter
from .router import Router, RoutingStrategy, TaskConfig, TaskType

# Alias for backward compatibility
Config = GatewayConfig

__all__ = [
    # Version
    "__version__",
    # Configuration
    "Config",  # ← ADD THIS (alias for GatewayConfig)
    "GatewayConfig",
    "ModelConfig",
    "config",
    # Models & Base
    "ModelHandler",
    "ModelResponse",
    "BaseLLM",
    "LocalRunner",
    "ProviderAdapter",
    # Router
    "Router",
    "TaskType",
    "TaskConfig",
    "RoutingStrategy",
    # Main
    "MultiMind",
    # Monitoring
    "ModelMonitor",
    "ModelMetrics",
    "ModelHealth",
    "monitor",
    # Chat
    "ChatManager",
    "ChatSession",
    "ChatMessage",
    "chat_manager",
    # Exceptions
    "ConfigurationError",
]
