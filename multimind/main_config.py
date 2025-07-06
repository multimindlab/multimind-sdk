"""
Configuration management for Multimind SDK.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from dotenv import load_dotenv

class Config:
    """Manages SDK configuration."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv("MULTIMIND_CONFIG")
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file and environment."""
        # Load environment variables
        load_dotenv()

        # Load config file if specified
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)

        # Override with environment variables
        self._load_env_vars()

    def _load_env_vars(self) -> None:
        """Load configuration from environment variables."""
        # OpenAI
        if "OPENAI_API_KEY" in os.environ:
            self.config.setdefault("openai", {})["api_key"] = os.environ["OPENAI_API_KEY"]

        # Anthropic
        if "ANTHROPIC_API_KEY" in os.environ:
            self.config.setdefault("anthropic", {})["api_key"] = os.environ["ANTHROPIC_API_KEY"]

        # Mistral
        if "MISTRAL_API_KEY" in os.environ:
            self.config.setdefault("mistral", {})["api_key"] = os.environ["MISTRAL_API_KEY"]

        # HuggingFace
        if "HUGGINGFACE_API_KEY" in os.environ:
            self.config.setdefault("huggingface", {})["api_key"] = os.environ["HUGGINGFACE_API_KEY"]

        # Ollama
        if "OLLAMA_HOST" in os.environ:
            self.config.setdefault("ollama", {})["host"] = os.environ["OLLAMA_HOST"]

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return defaul

            if value is None:
                return defaul

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            config = config.setdefault(k, {})

        config[keys[-1]] = value

    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file."""
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No config path specified")

        with open(save_path, 'w') as f:
            yaml.safe_dump(self.config, f)

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model type."""
        return self.get(f"models.{model_type}", {})

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        return self.get(f"{provider}.api_key")

    def get_model_params(self, model_type: str, model_name: str) -> Dict[str, Any]:
        """Get parameters for a specific model."""
        return self.get(f"models.{model_type}.{model_name}", {})