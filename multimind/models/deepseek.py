"""
DeepSeek model implementation via DeepSeek's OpenAI-compatible API.
"""

from typing import Dict, Optional, Tuple

from .openai import OpenAIModel


class DeepSeekModel(OpenAIModel):
    """DeepSeek model implementation (OpenAI-compatible endpoint)."""

    PROVIDER_NAME: str = "DeepSeek"
    API_KEY_ENV_VARS: Tuple[str, ...] = ("DEEPSEEK_API_KEY",)
    BASE_URL: Optional[str] = "https://api.deepseek.com/v1"
    DEFAULT_EMBEDDING_MODEL: Optional[str] = None
    MODEL_PRICING: Dict[str, float] = {}
    DEFAULT_COST_PER_TOKEN: Optional[float] = None
