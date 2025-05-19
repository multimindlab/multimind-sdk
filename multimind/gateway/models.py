"""
Model handlers for different AI providers
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

import openai
import anthropic
import requests
from groq import Groq
from huggingface_hub import InferenceClient

from .config import ModelConfig, config

logger = logging.getLogger(__name__)

@dataclass
class ModelResponse:
    """Standardized response from any model"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    timestamp: str = datetime.now().isoformat()

class ModelHandler(ABC):
    """Abstract base class for model handlers"""
    
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self._client = None
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Send a chat message to the model"""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate text from a prompt"""
        pass

class OpenAIHandler(ModelHandler):
    """Handler for OpenAI models"""
    
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self._client = openai.AsyncOpenAI(api_key=self.config.api_key)
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens)
            )
            
            return ModelResponse(
                content=response.choices[0].message.content,
                model=self.config.model_name,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                finish_reason=response.choices[0].finish_reason
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)

class AnthropicHandler(ModelHandler):
    """Handler for Anthropic models"""
    
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self._client = anthropic.AsyncAnthropic(api_key=self.config.api_key)
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        try:
            # Convert messages to Anthropic format
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
            response = await self._client.messages.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens)
            )
            
            return ModelResponse(
                content=response.content[0].text,
                model=self.config.model_name,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                finish_reason=response.stop_reason
            )
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)

class OllamaHandler(ModelHandler):
    """Handler for Ollama models"""
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        try:
            # Convert messages to Ollama format
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
            response = requests.post(
                f"{self.config.api_base}/api/generate",
                json={
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens)
                },
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return ModelResponse(
                content=result["response"],
                model=self.config.model_name,
                finish_reason=result.get("done", True) and "stop" or None
            )
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)

class GroqHandler(ModelHandler):
    """Handler for Groq models"""
    
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self._client = Groq(api_key=self.config.api_key)
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens)
            )
            
            return ModelResponse(
                content=response.choices[0].message.content,
                model=self.config.model_name,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                finish_reason=response.choices[0].finish_reason
            )
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)

class HuggingFaceHandler(ModelHandler):
    """Handler for HuggingFace models"""
    
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self._client = InferenceClient(
            model=self.config.model_name,
            token=self.config.api_key
        )
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        try:
            # Convert messages to prompt format
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
            response = await self._client.text_generation(
                prompt,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                return_full_text=False
            )
            
            return ModelResponse(
                content=response,
                model=self.config.model_name
            )
        except Exception as e:
            logger.error(f"HuggingFace API error: {str(e)}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)

def get_model_handler(model_name: str) -> ModelHandler:
    """Factory function to get the appropriate model handler"""
    model_map = {
        "openai": OpenAIHandler,
        "anthropic": AnthropicHandler,
        "ollama": OllamaHandler,
        "groq": GroqHandler,
        "huggingface": HuggingFaceHandler
    }
    
    handler_class = model_map.get(model_name.lower())
    if not handler_class:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model_config = config.get_model_config(model_name)
    return handler_class(model_config) 