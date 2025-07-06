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

__all__ = [
    "AdaptiveRouter",
    "FallbackHandler", 
    "MultiModalRouter",
    "ModelRouter",
    "RoutingStrategy",
    "CostAwareStrategy",
    "LatencyAwareStrategy", 
    "HybridStrategy"
] 