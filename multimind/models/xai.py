"""
xAI (Grok) model implementation via xAI's OpenAI-compatible API.
"""

from typing import Dict, Optional, Tuple

from .openai import OpenAIModel


class XAIModel(OpenAIModel):
    """xAI Grok model implementation (OpenAI-compatible endpoint)."""

    PROVIDER_NAME: str = "xAI"
    API_KEY_ENV_VARS: Tuple[str, ...] = ("XAI_API_KEY",)
    BASE_URL: Optional[str] = "https://api.x.ai/v1"
    DEFAULT_EMBEDDING_MODEL: Optional[str] = None
    MODEL_PRICING: Dict[str, float] = {}
    DEFAULT_COST_PER_TOKEN: Optional[float] = None
