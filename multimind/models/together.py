"""
Together AI model implementation via Together's OpenAI-compatible API.
"""

from typing import Dict, Optional, Tuple

from .openai import OpenAIModel


class TogetherModel(OpenAIModel):
    """Together AI model implementation (OpenAI-compatible endpoint)."""

    PROVIDER_NAME: str = "Together AI"
    API_KEY_ENV_VARS: Tuple[str, ...] = ("TOGETHER_API_KEY",)
    BASE_URL: Optional[str] = "https://api.together.xyz/v1"
    DEFAULT_EMBEDDING_MODEL: Optional[str] = "BAAI/bge-large-en-v1.5"
    MODEL_PRICING: Dict[str, float] = {}
    DEFAULT_COST_PER_TOKEN: Optional[float] = None
