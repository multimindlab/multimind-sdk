"""
Cerebras model implementation via Cerebras' OpenAI-compatible API.
"""

from typing import Dict, Optional, Tuple

from .openai import OpenAIModel


class CerebrasModel(OpenAIModel):
    """Cerebras model implementation (OpenAI-compatible endpoint)."""

    PROVIDER_NAME: str = "Cerebras"
    API_KEY_ENV_VARS: Tuple[str, ...] = ("CEREBRAS_API_KEY",)
    BASE_URL: Optional[str] = "https://api.cerebras.ai/v1"
    DEFAULT_EMBEDDING_MODEL: Optional[str] = None
    MODEL_PRICING: Dict[str, float] = {}
    DEFAULT_COST_PER_TOKEN: Optional[float] = None
