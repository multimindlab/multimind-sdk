"""
Main router interface for model selection and request routing.
"""

from typing import List, Dict, Any, Optional, Type
from ..models.base import BaseLLM
from .strategy import RoutingStrategy
from .fallback import FallbackHandler

class ModelRouter:
    """Routes requests to appropriate models with strategy and fallback support."""

    def __init__(self, strategy: Optional[RoutingStrategy] = None):
        self.models: Dict[str, BaseLLM] = {}
        self.strategy = strategy or RoutingStrategy()
        self.fallback = FallbackHandler()

    def register_model(self, name: str, model: BaseLLM) -> None:
        """Register a model with the router."""
        self.models[name] = model

    def set_strategy(self, strategy: RoutingStrategy) -> None:
        """Set the routing strategy."""
        self.strategy = strategy

    def set_fallback_chain(self, model_names: List[str]) -> None:
        """Set the fallback chain for model selection."""
        self.fallback.set_chain(model_names)

    async def get_model(
        self,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """Get a model instance based on strategy and fallback."""
        if model_name and model_name in self.models:
            return self.models[model_name]

        # Use strategy to select model
        selected_model = await self.strategy.select_model(
            list(self.models.values()),
            **kwargs
        )

        if selected_model:
            return selected_model

        # Fall back to fallback chain
        return await self.fallback.get_model(self.models)

    async def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using the appropriate model."""
        model = await self.get_model(model_name, **kwargs)
        try:
            return await model.generate(prompt, **kwargs)
        except Exception as e:
            if await self.fallback.should_retry(e):
                return await self.generate(prompt, **kwargs)
            raise

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate chat completion using the appropriate model."""
        model = await self.get_model(model_name, **kwargs)
        try:
            return await model.chat(messages, **kwargs)
        except Exception as e:
            if await self.fallback.should_retry(e):
                return await self.chat(messages, **kwargs)
            raise