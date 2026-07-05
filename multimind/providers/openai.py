"""
OpenAI provider adapter for the MultimindSDK.
"""

import base64
from datetime import datetime
from typing import Any, Dict, List, Optional

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..core.provider import (
    EmbeddingResult,
    GenerationResult,
    ImageAnalysisResult,
    ProviderAdapter,
    ProviderCapability,
    ProviderConfig,
    ProviderMetadata,
)


class OpenAIProvider(ProviderAdapter):
    """OpenAI provider adapter implementation."""

    def __init__(self, config: ProviderConfig):
        """Initialize the OpenAI provider adapter."""
        super().__init__(config)
        self.client = openai.AsyncOpenAI(api_key=config.api_key)

    @retry(
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APIError,
                openai.APIConnectionError,
                openai.APITimeoutError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True,
    )
    async def _chat_completions_create(self, **kwargs: Any):
        """Internal helper with retry for chat.completions.create."""
        return await self.client.chat.completions.create(**kwargs)

    @retry(
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APIError,
                openai.APIConnectionError,
                openai.APITimeoutError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True,
    )
    async def _embeddings_create(self, **kwargs: Any):
        """Internal helper with retry for embeddings.create."""
        return await self.client.embeddings.create(**kwargs)

    async def generate_text(
        self, prompt: str, model: str = "gpt-3.5-turbo", **kwargs
    ) -> GenerationResult:
        """Generate text using OpenAI's API."""
        start_time = datetime.now()

        try:
            response = await self._chat_completions_create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            result = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Calculate cost based on model pricing
            pricing = self.metadata.pricing.get(model, {"input": 0.0, "output": 0.0})
            if isinstance(pricing, dict):
                input_cost = pricing.get("input", 0.0)
                output_cost = pricing.get("output", 0.0)
            else:
                input_cost = 0.0
                output_cost = 0.0
            cost = (
                input_cost * response.usage.prompt_tokens
                + output_cost * response.usage.completion_tokens
            ) / 1000  # Convert to USD

            return GenerationResult(
                text=result,
                tokens_used=tokens_used,
                provider_name="openai",
                model_name=model,
                latency_ms=latency_ms,
                cost_estimate_usd=cost,
            )

        except openai.OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

    async def chat(
        self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", **kwargs
    ) -> GenerationResult:
        """Generate chat completion using OpenAI's API."""
        start_time = datetime.now()

        try:
            response = await self._chat_completions_create(
                model=model,
                messages=messages,
                **kwargs,
            )

            result = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Calculate cost based on model pricing
            pricing = self.metadata.pricing.get(model, {"input": 0.0, "output": 0.0})
            if isinstance(pricing, dict):
                input_cost = pricing.get("input", 0.0)
                output_cost = pricing.get("output", 0.0)
            else:
                input_cost = 0.0
                output_cost = 0.0
            cost = (
                input_cost * response.usage.prompt_tokens
                + output_cost * response.usage.completion_tokens
            ) / 1000  # Convert to USD

            return GenerationResult(
                text=result,
                tokens_used=tokens_used,
                provider_name="openai",
                model_name=model,
                latency_ms=latency_ms,
                cost_estimate_usd=cost,
            )

        except openai.OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

    async def generate_embeddings(
        self, text: str, model: str = "text-embedding-ada-002", **kwargs
    ) -> EmbeddingResult:
        """Generate embeddings using OpenAI's API."""
        start_time = datetime.now()

        try:
            response = await self._embeddings_create(
                model=model,
                input=text,
                **kwargs,
            )

            embedding_vector = response.data[0].embedding
            tokens_used = response.usage.total_tokens
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Calculate cost based on model pricing
            pricing = self.metadata.pricing.get(model, {"input": 0.0})
            cost = pricing["input"] * tokens_used / 1000  # Convert to USD

            return EmbeddingResult(
                provider_name="openai",
                model_name=model,
                embedding=embedding_vector,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost_estimate_usd=cost,
            )

        except openai.OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

    async def analyze_image(
        self, image_data: bytes, prompt: str, model: str = "gpt-4o-mini", **kwargs
    ) -> ImageAnalysisResult:
        """Analyze image using OpenAI's API."""
        start_time = datetime.now()

        try:
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            response = await self._chat_completions_create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                            },
                        ],
                    }
                ],
                **kwargs,
            )

            result = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Calculate cost based on model pricing
            pricing = self.metadata.pricing.get(model, {"input": 0.0, "output": 0.0})
            if isinstance(pricing, dict):
                input_cost = pricing.get("input", 0.0)
                output_cost = pricing.get("output", 0.0)
            else:
                input_cost = 0.0
                output_cost = 0.0
            cost = (
                input_cost * response.usage.prompt_tokens
                + output_cost * response.usage.completion_tokens
            ) / 1000  # Convert to USD

            return ImageAnalysisResult(
                objects=[],
                captions=[result] if result else [],
                text=result,
                provider_name="openai",
                model_name=model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost_estimate_usd=cost,
            )

        except openai.OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

    def _get_metadata(self) -> ProviderMetadata:
        """Return metadata about the OpenAI provider."""
        return ProviderMetadata(
            name="openai",
            version="1.0.0",
            capabilities=[
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.CHAT,
                ProviderCapability.EMBEDDINGS,
                ProviderCapability.IMAGE_ANALYSIS,
                ProviderCapability.CODE_GENERATION,
            ],
            pricing={
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
                "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
            },
            typical_latency_ms={"gpt-4": 500, "gpt-3.5-turbo": 200, "text-embedding-ada-002": 100},
            latency={"gpt-4": {"p50": 500, "p95": 1000}, "gpt-3.5-turbo": {"p50": 200, "p95": 400}},
            max_context_length=4096,
            max_tokens_per_request=2048,
            supported_models=["gpt-4", "gpt-3.5-turbo", "text-embedding-ada-002"],
        )

    async def estimate_cost(
        self, task_type: str, model: str, input_tokens: int, output_tokens: Optional[int] = None
    ) -> float:
        """Estimate cost for a given task."""
        pricing = self.metadata.pricing.get(model, {"input": 0.0, "output": 0.0})

        if task_type == "embeddings":
            return pricing["input"] * input_tokens / 1000
        else:
            return (
                pricing["input"] * input_tokens + pricing["output"] * (output_tokens or 0)
            ) / 1000

    async def estimate_latency(
        self, task_type: str, model: str, input_tokens: int, output_tokens: Optional[int] = None
    ) -> float:
        """Estimate latency for a given task."""
        if self.metadata.latency:
            latency = self.metadata.latency.get(model, {"p50": 0, "p95": 0})
            return latency["p50"]  # Return median latency
        return 0.0

    async def get_cost_estimate(
        self, operation: str, input_tokens: int, output_tokens: Optional[int] = None, **kwargs
    ) -> float:
        """Estimate cost for an operation (abstract method implementation)."""
        # Extract model from kwargs or use default
        model = kwargs.get("model", "gpt-3.5-turbo")
        pricing = self.metadata.pricing.get(model, {"input": 0.0, "output": 0.0})

        if operation == "embeddings":
            return pricing["input"] * input_tokens / 1000
        else:
            return (
                pricing["input"] * input_tokens + pricing["output"] * (output_tokens or 0)
            ) / 1000

    async def get_latency_estimate(self, operation: str, **kwargs) -> float:
        """Estimate latency for an operation (abstract method implementation)."""
        # Extract model from kwargs or use default
        model = kwargs.get("model", "gpt-3.5-turbo")
        if self.metadata.latency:
            latency = self.metadata.latency.get(model, {"p50": 0, "p95": 0})
            return latency["p50"]  # Return median latency
        return 0.0
