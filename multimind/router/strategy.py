"""
Routing strategies for model selection based on cost and latency.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..models.base import BaseLLM

class RoutingStrategy(ABC):
    """Abstract base class for routing strategies."""

    @abstractmethod
    async def select_model(
        self,
        models: List[BaseLLM],
        **kwargs
    ) -> Optional[BaseLLM]:
        """Select a model based on the strategy."""
        pass

class CostAwareStrategy(RoutingStrategy):
    """Selects model based on cost per token."""

    async def select_model(
        self,
        models: List[BaseLLM],
        prompt_tokens: int,
        max_completion_tokens: int,
        **kwargs
    ) -> Optional[BaseLLM]:
        """Select the model with lowest expected cost."""
        if not models:
            return None

        min_cost = float('inf')
        selected_model = None

        for model in models:
            cost = await model.get_cost(prompt_tokens, max_completion_tokens)
            if cost < min_cost:
                min_cost = cos
                selected_model = model

        return selected_model

class LatencyAwareStrategy(RoutingStrategy):
    """Selects model based on latency."""

    async def select_model(
        self,
        models: List[BaseLLM],
        **kwargs
    ) -> Optional[BaseLLM]:
        """Select the model with lowest latency."""
        if not models:
            return None

        min_latency = float('inf')
        selected_model = None

        for model in models:
            latency = await model.get_latency()
            if latency is not None and latency < min_latency:
                min_latency = latency
                selected_model = model

        return selected_model

class HybridStrategy(RoutingStrategy):
    """Combines cost and latency awareness."""

    def __init__(self, cost_weight: float = 0.5, latency_weight: float = 0.5):
        self.cost_weight = cost_weigh
        self.latency_weight = latency_weigh

    async def select_model(
        self,
        models: List[BaseLLM],
        prompt_tokens: int,
        max_completion_tokens: int,
        **kwargs
    ) -> Optional[BaseLLM]:
        """Select model based on weighted cost and latency."""
        if not models:
            return None

        best_score = float('inf')
        selected_model = None

        for model in models:
            cost = await model.get_cost(prompt_tokens, max_completion_tokens)
            latency = await model.get_latency() or float('inf')

            # Normalize and combine scores
            cost_score = cost * self.cost_weigh
            latency_score = latency * self.latency_weigh
            total_score = cost_score + latency_score

            if total_score < best_score:
                best_score = total_score
                selected_model = model

        return selected_model