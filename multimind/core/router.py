"""
Core router for model selection and request routing.
"""

from typing import Dict, Any, Optional, Lis
from ..models.base import BaseLLM
from ..models.factory import ModelFactory

class ModelRouter:
    """Routes requests to appropriate models with fallback support."""

    def __init__(self, env_path: Optional[str] = None):
        self.factory = ModelFactory(env_path=env_path)
        self.fallback_chain: List[str] = []

    def available_models(self) -> List[str]:
        """Get list of available models."""
        return self.factory.available_models()

    def set_fallback_chain(self, providers: List[str]) -> None:
        """Set the fallback chain for model selection."""
        available = self.available_models()
        for provider in providers:
            if provider not in available:
                raise ValueError(f"Provider {provider} is not available")
        self.fallback_chain = providers

    async def get_model(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """Get a model instance, following the fallback chain if needed."""
        if provider:
            return self.factory.get_model(provider, model_name, **kwargs)

        # Try fallback chain
        for fallback_provider in self.fallback_chain:
            try:
                return self.factory.get_model(fallback_provider, model_name, **kwargs)
            except Exception:
                continue

        raise ValueError("No available models in the fallback chain")

    async def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using the appropriate model."""
        model = await self.get_model(provider, model_name)
        return await model.generate(prompt, **kwargs)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate chat completion using the appropriate model."""
        model = await self.get_model(provider, model_name)
        return await model.chat(messages, **kwargs)

    async def embeddings(
        self,
        text: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> List[float]:
        """Generate embeddings using the appropriate model."""
        model = await self.get_model(provider, model_name)
        return await model.embeddings(text, **kwargs)