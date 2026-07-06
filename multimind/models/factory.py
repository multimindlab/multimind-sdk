"""
Factory for creating and managing model instances.
"""

import os
from typing import Dict, Final, List, Optional, Type

from dotenv import load_dotenv

from ..core.exceptions import ConfigurationError
from .base import BaseLLM
from .claude import ClaudeModel
from .deepseek import DeepSeekModel
from .gemini import GeminiModel
from .groq import GroqModel
from .mistral import MistralAIModel
from .ollama import OllamaModel
from .openai import OpenAIModel


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
            "ollama": OllamaModel,
            "groq": GroqModel,
            "mistral": MistralAIModel,
            "gemini": GeminiModel,
            "deepseek": DeepSeekModel,
        }

        # Initialize API keys
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.claude_key = os.getenv("CLAUDE_API_KEY")

    def available_models(self) -> List[str]:
        """Get list of available model providers based on API keys."""
        available = []

        # Check API keys
        if self.openai_key:
            available.append("openai")
        if self.claude_key:
            available.append("claude")
        if os.getenv("GROQ_API_KEY"):
            available.append("groq")
        if os.getenv("MISTRAL_API_KEY"):
            available.append("mistral")
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            available.append("gemini")
        if os.getenv("DEEPSEEK_API_KEY"):
            available.append("deepseek")

        # Check Ollama availability (server + client libs)
        OLLAMA_HOST: Final[str] = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        try:
            import aiohttp

            async def _check_ollama() -> bool:
                timeout = aiohttp.ClientTimeout(total=2)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    try:
                        async with session.get(f"{OLLAMA_HOST}/api/tags") as resp:
                            return resp.status == 200
                    except aiohttp.ClientError:
                        return False

            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            is_available = False
            if loop and loop.is_running():
                # Best-effort: schedule check in running loop and assume available
                loop.create_task(_check_ollama())
                is_available = True
            else:
                is_available = asyncio.run(_check_ollama())

            if is_available:
                available.append("ollama")
        except ImportError:
            # aiohttp not installed, treat Ollama as unavailable
            pass

        return available

    def get_model(self, provider: str, model_name: Optional[str] = None, **kwargs) -> BaseLLM:
        """Get or create a model instance."""
        if provider not in self._model_classes:
            raise ValueError(f"Unsupported model provider: {provider}")

        # Use default model names if not specified
        if model_name is None:
            model_name = {
                "openai": "gpt-4",
                "claude": "claude-3-opus-20240229",
                "ollama": "mistral",
                "groq": "llama-3.3-70b-versatile",
                "mistral": "mistral-small-latest",
                "gemini": "gemini-2.0-flash",
                "deepseek": "deepseek-chat",
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
            api_key = kwargs.get("api_key", self.openai_key)
            if not api_key:
                raise ConfigurationError(
                    "OpenAI API key is not configured. "
                    "Set the OPENAI_API_KEY environment variable or pass api_key explicitly."
                )
            kwargs["api_key"] = api_key
        elif provider == "claude":
            api_key = kwargs.get("api_key", self.claude_key)
            if not api_key:
                raise ConfigurationError(
                    "Claude API key is not configured. "
                    "Set ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable, "
                    "or pass api_key explicitly."
                )
            kwargs["api_key"] = api_key

        # Create and store instance
        instance = model_class(model_name=model_name, **kwargs)
        self._instances[instance_key] = instance

        return instance

    def register_model_class(self, provider: str, model_class: Type[BaseLLM]) -> None:
        """Register a new model class."""
        self._model_classes[provider] = model_class
