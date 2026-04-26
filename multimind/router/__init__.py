"""
Router module for MultiMind SDK.

This module provides routing capabilities for directing requests to appropriate models
and handling fallback strategies.
"""

from .adaptive import AdaptiveRouter
from .fallback import FallbackHandler
from .multi_modal_router import MultiModalRouter
from .router import ModelRouter
from .strategy import RoutingStrategy, CostAwareStrategy, LatencyAwareStrategy, HybridStrategy

# Import Router from core (fix the circular import issue)
try:
    from ..core.router import Router, TaskType, TaskConfig
except ImportError:
    # Fallback if core router not available
    Router = ModelRouter
    TaskType = None
    TaskConfig = None

__all__ = [
    "AdaptiveRouter",
    "FallbackHandler", 
    "MultiModalRouter",
    "ModelRouter",
    "Router",
    "TaskType",
    "TaskConfig",
    "RoutingStrategy",
    "CostAwareStrategy",
    "LatencyAwareStrategy", 
    "HybridStrategy"
]
