"""
Core configuration functionality for MultiMind
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Optional pydantic-settings import
try:
    from pydantic_settings import BaseSettings

    PYDANTIC_SETTINGS_AVAILABLE = True
except ImportError:
    PYDANTIC_SETTINGS_AVAILABLE = False
    logger.warning("pydantic-settings not available. Configuration features will be disabled.")
    # Fallback to pydantic BaseModel
    from pydantic import BaseModel as BaseSettings

from dotenv import load_dotenv
from pydantic import Field

# Load environment variables from .env file
load_dotenv()


class ModelConfig(BaseSettings):
    """Configuration for individual models"""

    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: int = 30


class GatewayConfig(BaseSettings):
    """Main configuration for the MultiMind Gateway"""

    # OpenAI Configuration
    openai: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo"),
        )
    )

    # Anthropic Configuration
    anthropic: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model_name=os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-opus-20240229"),
        )
    )

    # Ollama Configuration
    ollama: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
            model_name=os.getenv("OLLAMA_MODEL_NAME", "mistral"),
        )
    )

    # Groq Configuration
    groq: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL_NAME", "mixtral-8x7b-32768"),
        )
    )

    # HuggingFace Configuration
    huggingface: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("HUGGINGFACE_API_KEY"),  # Optional - if None, uses local transformers
            model_name=os.getenv(
                "HUGGINGFACE_MODEL_NAME", "gpt2"
            ),  # Default to small model for local testing
        )
    )

    gemini: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash"),
        )
    )
    mistral: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model_name=os.getenv("MISTRAL_MODEL_NAME", "mistral-small-latest"),
        )
    )
    deepseek: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            model_name=os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat"),
        )
    )
    openrouter: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model_name=os.getenv("OPENROUTER_MODEL_NAME", "openai/gpt-4o-mini"),
        )
    )
    together: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("TOGETHER_API_KEY"),
            model_name=os.getenv(
                "TOGETHER_MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
            ),
        )
    )
    xai: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("XAI_API_KEY"),
            model_name=os.getenv("XAI_MODEL_NAME", "grok-2-latest"),
        )
    )
    perplexity: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            model_name=os.getenv("PERPLEXITY_MODEL_NAME", "sonar"),
        )
    )
    fireworks: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("FIREWORKS_API_KEY"),
            model_name=os.getenv(
                "FIREWORKS_MODEL_NAME", "accounts/fireworks/models/llama-v3p1-8b-instruct"
            ),
        )
    )
    cerebras: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("CEREBRAS_API_KEY"),
            model_name=os.getenv("CEREBRAS_MODEL_NAME", "llama3.1-8b"),
        )
    )
    claude: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY"),
            model_name=os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-5-sonnet-20241022"),
        )
    )

    # General Settings
    default_model: str = Field(
        default=os.getenv("DEFAULT_MODEL", "openai"),
        description="Default model to use when none specified",
    )

    log_level: str = Field(
        default=os.getenv("LOG_LEVEL", "INFO"), description="Logging level for the gateway"
    )

    model_config = {"env_prefix": "MULTIMIND_", "case_sensitive": False}

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model"""
        model_map = {
            "openai": self.openai,
            "anthropic": self.anthropic,
            "claude": self.claude,
            "ollama": self.ollama,
            "groq": self.groq,
            "huggingface": self.huggingface,
            "gemini": self.gemini,
            "mistral": self.mistral,
            "deepseek": self.deepseek,
            "openrouter": self.openrouter,
            "together": self.together,
            "xai": self.xai,
            "perplexity": self.perplexity,
            "fireworks": self.fireworks,
            "cerebras": self.cerebras,
        }
        normalized_name = model_name.lower()
        if normalized_name not in model_map:
            available_models = ", ".join(sorted(model_map.keys()))
            raise ValueError(
                f"Unknown model name '{model_name}'. Available models: {available_models}"
            )
        return model_map[normalized_name]

    @classmethod
    def validate(cls, value: "GatewayConfig") -> "GatewayConfig":
        """
        Pydantic hook to validate the full GatewayConfig instance.
        Ensures that the configured default_model refers to a known model section.
        """
        allowed_models = {
            "openai",
            "anthropic",
            "claude",
            "ollama",
            "groq",
            "huggingface",
            "gemini",
            "mistral",
            "deepseek",
            "openrouter",
            "together",
            "xai",
            "perplexity",
            "fireworks",
            "cerebras",
        }
        normalized_default = (value.default_model or "").lower()
        if normalized_default not in allowed_models:
            available = ", ".join(sorted(allowed_models))
            raise ValueError(
                f"Invalid default_model '{value.default_model}'. Must be one of: {available}"
            )
        return value


# Create a global config instance
config = GatewayConfig()
