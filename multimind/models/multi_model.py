"""
MultiModelWrapper for managing multiple model instances with fallback support.
"""

from typing import List, Dict, Optional, Union, AsyncGenerator
import asyncio
import logging
from .base import BaseLLM
from .factory import ModelFactory

logger = logging.getLogger(__name__)

class MultiModelWrapper:
    """Wrapper for managing multiple model instances with fallback support."""
    
    def __init__(
        self,
        model_factory: ModelFactory,
        primary_model: str = "openai",
        fallback_models: Optional[List[str]] = None,
        model_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the MultiModelWrapper.
        
        Args:
            model_factory: Factory for creating model instances
            primary_model: Primary model to use
            fallback_models: List of fallback models to try if primary fails
            model_weights: Optional weights for model selection
        """
        self.model_factory = model_factory
        self.primary_model = primary_model
        self.fallback_models = fallback_models or []
        self.model_weights = model_weights or {}
        
        # Initialize models
        self._models: Dict[str, BaseLLM] = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all required model instances."""
        all_models = [self.primary_model] + self.fallback_models
        
        for model_name in all_models:
            try:
                self._models[model_name] = self.model_factory.get_model(model_name)
                logger.info(f"Initialized model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize model {model_name}: {e}")
    
    def _get_model(self, model_name: str) -> Optional[BaseLLM]:
        """Get a model instance by name."""
        return self._models.get(model_name)
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models."""
        return [name for name, model in self._models.items() if model is not None]
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text using the primary model with fallback support.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text
        """
        models_to_try = [self.primary_model] + self.fallback_models
        
        for model_name in models_to_try:
            model = self._get_model(model_name)
            if model is None:
                continue
                
            try:
                logger.info(f"Attempting generation with model: {model_name}")
                result = await model.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                logger.info(f"Successfully generated with model: {model_name}")
                return result
                
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                continue
        
        # If all models failed
        raise RuntimeError("All models failed to generate text")
    
    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate text stream using the primary model with fallback support.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Yields:
            Generated text chunks
        """
        models_to_try = [self.primary_model] + self.fallback_models
        
        for model_name in models_to_try:
            model = self._get_model(model_name)
            if model is None:
                continue
                
            try:
                logger.info(f"Attempting stream generation with model: {model_name}")
                async for chunk in model.generate_stream(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                ):
                    yield chunk
                logger.info(f"Successfully streamed with model: {model_name}")
                return
                
            except Exception as e:
                logger.warning(f"Model {model_name} failed for streaming: {e}")
                continue
        
        # If all models failed
        raise RuntimeError("All models failed to generate text stream")
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate chat completion using the primary model with fallback support.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            Chat completion text
        """
        models_to_try = [self.primary_model] + self.fallback_models
        
        for model_name in models_to_try:
            model = self._get_model(model_name)
            if model is None:
                continue
                
            try:
                logger.info(f"Attempting chat with model: {model_name}")
                result = await model.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                logger.info(f"Successfully chatted with model: {model_name}")
                return result
                
            except Exception as e:
                logger.warning(f"Model {model_name} failed for chat: {e}")
                continue
        
        # If all models failed
        raise RuntimeError("All models failed to generate chat completion")
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate chat completion stream using the primary model with fallback support.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Yields:
            Chat completion text chunks
        """
        models_to_try = [self.primary_model] + self.fallback_models
        
        for model_name in models_to_try:
            model = self._get_model(model_name)
            if model is None:
                continue
                
            try:
                logger.info(f"Attempting chat stream with model: {model_name}")
                async for chunk in model.chat_stream(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                ):
                    yield chunk
                logger.info(f"Successfully streamed chat with model: {model_name}")
                return
                
            except Exception as e:
                logger.warning(f"Model {model_name} failed for chat streaming: {e}")
                continue
        
        # If all models failed
        raise RuntimeError("All models failed to generate chat completion stream")
    
    async def embeddings(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings using the primary model with fallback support.
        
        Args:
            text: Input text or list of texts
            **kwargs: Additional model-specific parameters
            
        Returns:
            Embeddings as list of floats or list of lists of floats
        """
        models_to_try = [self.primary_model] + self.fallback_models
        
        for model_name in models_to_try:
            model = self._get_model(model_name)
            if model is None:
                continue
                
            try:
                logger.info(f"Attempting embeddings with model: {model_name}")
                result = await model.embeddings(text, **kwargs)
                logger.info(f"Successfully generated embeddings with model: {model_name}")
                return result
                
            except Exception as e:
                logger.warning(f"Model {model_name} failed for embeddings: {e}")
                continue
        
        # If all models failed
        raise RuntimeError("All models failed to generate embeddings")
    
    def get_model_info(self) -> Dict[str, Dict]:
        """Get information about all available models."""
        info = {}
        for name, model in self._models.items():
            if model is not None:
                info[name] = {
                    "model_name": model.model_name,
                    "capabilities": model.get_capabilities(),
                    "cost_per_token": model.cost_per_token,
                    "avg_latency": model.avg_latency
                }
        return info
    
    def get_primary_model(self) -> str:
        """Get the primary model name."""
        return self.primary_model
    
    def get_fallback_models(self) -> List[str]:
        """Get the fallback models list."""
        return self.fallback_models.copy()
    
    def add_fallback_model(self, model_name: str):
        """Add a new fallback model."""
        if model_name not in self.fallback_models:
            self.fallback_models.append(model_name)
            try:
                self._models[model_name] = self.model_factory.get_model(model_name)
                logger.info(f"Added fallback model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to add fallback model {model_name}: {e}")
    
    def remove_fallback_model(self, model_name: str):
        """Remove a fallback model."""
        if model_name in self.fallback_models:
            self.fallback_models.remove(model_name)
            if model_name in self._models:
                del self._models[model_name]
            logger.info(f"Removed fallback model: {model_name}")

