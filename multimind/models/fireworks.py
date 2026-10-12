"""
Fireworks AI model implementation via Fireworks' OpenAI-compatible API.
"""

from typing import Dict, Optional, Tuple

from .openai import OpenAIModel


class FireworksModel(OpenAIModel):
    """Fireworks AI model implementation (OpenAI-compatible endpoint)."""

    PROVIDER_NAME: str = "Fireworks AI"
    API_KEY_ENV_VARS: Tuple[str, ...] = ("FIREWORKS_API_KEY",)
    BASE_URL: Optional[str] = "https://api.fireworks.ai/inference/v1"
    DEFAULT_EMBEDDING_MODEL: Optional[str] = "nomic-ai/nomic-embed-text-v1.5"
    MODEL_PRICING: Dict[str, float] = {}
    DEFAULT_COST_PER_TOKEN: Optional[float] = None
