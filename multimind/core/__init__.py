"""
MultiMind Core - Shared functionality for the MultiMind project
"""

__version__ = "0.1.0"

from .models import ModelHandler, ModelResponse
from .config import GatewayConfig, ModelConfig, config
from .monitoring import ModelMonitor, ModelMetrics, ModelHealth, monitor
from .chat import ChatManager, ChatSession, ChatMessage, chat_manager
from .base import BaseLLM
from .router import Router, TaskType, TaskConfig, RoutingStrategy
from .multimind import MultiMind
from .local_runner import LocalRunner
from .provider import ProviderAdapter
from .exceptions import ConfigurationError

# Alias for backward compatibility
Config = GatewayConfig

__all__ = [
    # Version
    "__version__",
    
    # Configuration
    "Config",           # ← ADD THIS (alias for GatewayConfig)
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