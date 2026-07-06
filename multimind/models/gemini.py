"""
Google Gemini model implementation via Gemini's OpenAI-compatible API.
"""

from typing import Dict, Optional, Tuple

from .openai import OpenAIModel


class GeminiModel(OpenAIModel):
    """Google Gemini model implementation (OpenAI-compatible endpoint)."""

    PROVIDER_NAME: str = "Gemini"
    API_KEY_ENV_VARS: Tuple[str, ...] = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    BASE_URL: Optional[str] = "https://generativelanguage.googleapis.com/v1beta/openai/"
    DEFAULT_EMBEDDING_MODEL: Optional[str] = "gemini-embedding-001"
    MODEL_PRICING: Dict[str, float] = {}
    DEFAULT_COST_PER_TOKEN: Optional[float] = None
