"""
Fallback handler for managing model fallbacks and retries.
"""

from typing import List, Dict, Any, Optional, Type
from ..models.base import BaseLLM

class FallbackHandler:
    """Handles model fallbacks and retries."""

    def __init__(self, max_retries: int = 3):
        self.fallback_chain: List[str] = []
        self.max_retries = max_retries
        self.retry_count = 0
        self.retryable_errors = {
            "rate_limit_exceeded",
            "timeout",
            "service_unavailable",
            "internal_server_error"
        }

    def set_chain(self, model_names: List[str]) -> None:
        """Set the fallback chain for model selection."""
        self.fallback_chain = model_names

    async def get_model(self, models: Dict[str, BaseLLM]) -> BaseLLM:
        """Get the next model in the fallback chain."""
        for name in self.fallback_chain:
            if name in models:
                return models[name]
        raise ValueError("No available models in the fallback chain")

    async def should_retry(self, error: Exception) -> bool:
        """Determine if a request should be retried."""
        if self.retry_count >= self.max_retries:
            return False

        error_type = type(error).__name__.lower()
        if error_type in self.retryable_errors:
            self.retry_count += 1
            return True

        return False

    def reset(self) -> None:
        """Reset the retry counter."""
        self.retry_count = 0