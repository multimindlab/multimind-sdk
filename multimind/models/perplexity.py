"""
Perplexity model implementation via Perplexity's OpenAI-compatible API.
"""

from typing import Dict, Optional, Tuple

from .openai import OpenAIModel


class PerplexityModel(OpenAIModel):
    """Perplexity (Sonar) model implementation (OpenAI-compatible endpoint)."""

    PROVIDER_NAME: str = "Perplexity"
    API_KEY_ENV_VARS: Tuple[str, ...] = ("PERPLEXITY_API_KEY",)
    BASE_URL: Optional[str] = "https://api.perplexity.ai"
    DEFAULT_EMBEDDING_MODEL: Optional[str] = None
    MODEL_PRICING: Dict[str, float] = {}
    DEFAULT_COST_PER_TOKEN: Optional[float] = None
