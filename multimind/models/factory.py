"""
Factory for creating and managing model instances.
"""

import os
from typing import Dict, Optional, List, Type
from dotenv import load_dotenv

from .base import BaseLLM
from .openai import OpenAIModel
from .claude import ClaudeModel
from .ollama import OllamaModel

class ModelFactory:
    """Factory for creating and managing model instances."""
    
    def __init__(self, env_path: Optional[str] = None):
        # Load environment variables
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
            
        # Store model instances
        self._instances: Dict[str, BaseLLM] = {}
        
        # Model class mappings
        self._model_classes: Dict[str, Type[BaseLLM]] = {
            "openai": OpenAIModel,
            "claude": ClaudeModel,
            "ollama": OllamaModel
        }
        
        # Initialize API keys
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.claude_key = os.getenv('CLAUDE_API_KEY')
        
    def available_models(self) -> List[str]:
        """Get list of available model providers based on API keys."""
        available = []
        
        # Check API keys
        if self.openai_key:
            available.append("openai")
        if self.claude_key:
            available.append("claude")
            
        # Check Ollama availability
        try:
            import aiohttp
            available.append("ollama")
        except ImportError:
            pass
            
        return available
        
    def get_model(
        self,
        provider: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """Get or create a model instance."""
        if provider not in self._model_classes:
            raise ValueError(f"Unsupported model provider: {provider}")
            
        # Use default model names if not specified
        if model_name is None:
            model_name = {
                "openai": "gpt-4",
                "claude": "claude-3-opus-20240229",
                "ollama": "mistral"
            }.get(provider)
            
        # Create instance key
        instance_key = f"{provider}:{model_name}"
        
        # Return existing instance if available
        if instance_key in self._instances:
            return self._instances[instance_key]
            
        # Create new instance
        model_class = self._model_classes[provider]
        
        # Add API keys if needed
        if provider == "openai":
            kwargs["api_key"] = kwargs.get("api_key", self.openai_key)
        elif provider == "claude":
            kwargs["api_key"] = kwargs.get("api_key", self.claude_key)
            
        # Create and store instance
        instance = model_class(model_name=model_name, **kwargs)
        self._instances[instance_key] = instance
        
        return instance
        
    def register_model_class(self, provider: str, model_class: Type[BaseLLM]) -> None:
        """Register a new model class."""
        self._model_classes[provider] = model_class 