"""
Mistral AI model implementation via Mistral's OpenAI-compatible API.

Distinct from ``multimind.models.ollama.MistralModel``, which runs Mistral
models locally through Ollama.
"""

from typing import Dict, Optional, Tuple

from .openai import OpenAIModel


class MistralAIModel(OpenAIModel):
    """Mistral AI (La Plateforme) model implementation (OpenAI-compatible endpoint)."""

    PROVIDER_NAME: str = "Mistral AI"
    API_KEY_ENV_VARS: Tuple[str, ...] = ("MISTRAL_API_KEY",)
    BASE_URL: Optional[str] = "https://api.mistral.ai/v1"
    DEFAULT_EMBEDDING_MODEL: Optional[str] = "mistral-embed"
    MODEL_PRICING: Dict[str, float] = {}
    DEFAULT_COST_PER_TOKEN: Optional[float] = None
