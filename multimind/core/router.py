"""
Model router for handling model selection and fallback logic.
"""

from typing import List, Dict, Any, Optional, Type
from .base import BaseLLM

class ModelRouter:
    """Routes requests to appropriate models with fallback support."""
    
    def __init__(self):
        self.models: Dict[str, BaseLLM] = {}
        self.fallback_chain: List[str] = []
        
    def register_model(self, name: str, model: BaseLLM) -> None:
        """Register a model with the router."""
        self.models[name] = model
        
    def set_fallback_chain(self, model_names: List[str]) -> None:
        """Set the fallback chain for model selection."""
        self.fallback_chain = model_names
        
    async def get_model(self, model_name: Optional[str] = None) -> BaseLLM:
        """Get a model instance, following the fallback chain if needed."""
        if model_name and model_name in self.models:
            return self.models[model_name]
            
        for name in self.fallback_chain:
            if name in self.models:
                return self.models[name]
                
        raise ValueError("No available models in the fallback chain")
        
    async def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using the appropriate model."""
        model = await self.get_model(model_name)
        return await model.generate(prompt, **kwargs)
        
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate chat completion using the appropriate model."""
        model = await self.get_model(model_name)
        return await model.chat(messages, **kwargs) 