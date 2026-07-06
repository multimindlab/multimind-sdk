"""
OpenRouter model implementation via OpenRouter's OpenAI-compatible API.
"""

from typing import Dict, Optional, Tuple

from .openai import OpenAIModel


class OpenRouterModel(OpenAIModel):
    """OpenRouter aggregator (OpenAI-compatible endpoint).

    A single OPENROUTER_API_KEY exposes 300+ models from dozens of providers
    (OpenAI, Anthropic, Google, Meta, Mistral, and more) through one endpoint.
    Pass any OpenRouter model id (e.g. "anthropic/claude-sonnet-4") as model_name.
    """

    PROVIDER_NAME: str = "OpenRouter"
    API_KEY_ENV_VARS: Tuple[str, ...] = ("OPENROUTER_API_KEY",)
    BASE_URL: Optional[str] = "https://openrouter.ai/api/v1"
    DEFAULT_EMBEDDING_MODEL: Optional[str] = None
    # Pricing varies per routed model; pass cost_per_token explicitly if tracking cost.
    MODEL_PRICING: Dict[str, float] = {}
    DEFAULT_COST_PER_TOKEN: Optional[float] = None
