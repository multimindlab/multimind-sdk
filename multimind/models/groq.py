"""
Groq model implementation via Groq's OpenAI-compatible API.
"""

from typing import Dict, Optional, Tuple

from .openai import OpenAIModel


class GroqModel(OpenAIModel):
    """Groq model implementation (OpenAI-compatible endpoint)."""

    PROVIDER_NAME: str = "Groq"
    API_KEY_ENV_VARS: Tuple[str, ...] = ("GROQ_API_KEY",)
    BASE_URL: Optional[str] = "https://api.groq.com/openai/v1"
    DEFAULT_EMBEDDING_MODEL: Optional[str] = None
    MODEL_PRICING: Dict[str, float] = {}
    DEFAULT_COST_PER_TOKEN: Optional[float] = None
