"""
Ollama provider adapter for the MultimindSDK.
"""

import asyncio
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import aiohttp
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


class OllamaProvider(ProviderAdapter):
    """Ollama provider adapter implementation for local models."""

    def __init__(self, config: ProviderConfig):
        """Initialize the Ollama provider adapter."""
        super().__init__(config)
        # Default to localhost:11434 if no base URL is provided
        self.base_url = (config.api_base or "http://localhost:11434").rstrip("/")
        # For local models, use a longer default timeout (600s = 10 minutes)
        # Local models on CPU can be slow, especially for large requests
        import os

        timeout_override = os.getenv("OLLAMA_TIMEOUT")
        default_timeout = int(timeout_override) if timeout_override else 600
        # If no timeout explicitly configured, fall back to default_timeout
        self.config.timeout = getattr(self.config, "timeout", None) or default_timeout

    async def _make_request(
        self, endpoint: str, data: Dict[str, Any], timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make a request to the Ollama API."""
        url = f"{self.base_url}/{endpoint}"
        # Use provided timeout or default from config
        # For Ollama, default timeout is longer (600s) since local models can be slow on CPU
        request_timeout = timeout or getattr(self.config, "timeout", 600)
        return await self._make_request_with_retry(url, data, request_timeout)

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True,
    )
    async def _make_request_with_retry(
        self,
        url: str,
        data: Dict[str, Any],
        request_timeout: int,
    ) -> Dict[str, Any]:
        """Low-level JSON request with retry for connection issues."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=request_timeout),
                ) as response:
                    if response.status != 200:
                        # Try to parse as JSON first for better error messages
                        try:
                            error_json = await response.json()
                            error_msg = error_json.get("error", str(error_json))
                        except Exception:
                            # If not JSON, get text
                            error_text = await response.text()
                            error_msg = error_text or f"HTTP {response.status}"
                        raise Exception(f"Ollama API error ({response.status}): {error_msg}")
                    try:
                        return await response.json()
                    except Exception:
                        # If response is not valid JSON, try to get text for debugging
                        text_response = await response.text()
                        raise Exception(f"Invalid JSON response from Ollama: {text_response[:200]}")
        except asyncio.TimeoutError:
            raise Exception(
                f"Ollama request timeout after {request_timeout} seconds. Operations can take longer on CPU - consider using GPU or increasing timeout."
            )
        except aiohttp.ClientError as e:
            error_msg = str(e) if e else repr(e)
            raise Exception(f"Ollama connection error: {error_msg}")
        except Exception as e:
            # If it's already an Ollama error, re-raise it
            error_str = str(e) if e else repr(e)
            if (
                "Ollama API error" in error_str
                or "Ollama connection error" in error_str
                or "Ollama request timeout" in error_str
            ):
                raise
            # Otherwise, wrap it with more context
            if not error_str or error_str.strip() == "":
                error_str = f"Unknown error: {type(e).__name__}"
            raise Exception(f"Ollama request error: {error_str}")

    async def generate_text(self, prompt: str, model: str = "llama2", **kwargs) -> GenerationResult:
        """Generate text using Ollama's API."""
        start_time = datetime.now()

        try:
            data = {"model": model, "prompt": prompt, "stream": False, **kwargs}

            # Text generation can take time on CPU, use 5 minute timeout
            response = await self._make_request("api/generate", data)

            result = response.get("response", "")
            # Ollama doesn't always provide token counts, so we estimate
            tokens_used = response.get("eval_count", 0) + response.get("prompt_eval_count", 0)
            if tokens_used == 0:
                # Rough estimation: ~4 characters per token
                tokens_used = max(1, len(prompt + result) // 4)

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Ollama is free (local), so cost is 0
            return GenerationResult(
                text=result,
                tokens_used=tokens_used,
                provider_name="ollama",
                model_name=model,
                latency_ms=latency_ms,
                cost_estimate_usd=0.0,
            )

        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")

    async def chat(
        self, messages: List[Dict[str, str]], model: str = "llama2", **kwargs
    ) -> GenerationResult:
        """Generate chat completion using Ollama's API."""
        start_time = datetime.now()

        try:
            data = {"model": model, "messages": messages, "stream": False, **kwargs}

            # Chat completion can take time on CPU, use 5 minute timeout
            response = await self._make_request("api/chat", data)

            result = response.get("message", {}).get("content", "")
            # Ollama doesn't always provide token counts, so we estimate
            tokens_used = response.get("eval_count", 0) + response.get("prompt_eval_count", 0)
            if tokens_used == 0:
                # Rough estimation: ~4 characters per token
                total_text = " ".join([msg.get("content", "") for msg in messages]) + result
                tokens_used = max(1, len(total_text) // 4)

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Ollama is free (local), so cost is 0
            return GenerationResult(
                text=result,
                tokens_used=tokens_used,
                provider_name="ollama",
                model_name=model,
                latency_ms=latency_ms,
                cost_estimate_usd=0.0,
            )

        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")

    async def generate_embeddings(
        self, text: Union[str, List[str]], model: str = "llama2", **kwargs
    ) -> EmbeddingResult:
        """Generate embeddings using Ollama's API."""
        start_time = datetime.now()

        try:
            # Ollama embeddings endpoint expects a single prompt string.
            # If a list is provided, concatenate all texts so none are silently dropped.
            if isinstance(text, list):
                text_input = "\n\n".join(text)
            else:
                text_input = text

            data = {"model": model, "prompt": text_input, **kwargs}

            # Embeddings are usually faster, but use 2 minute timeout to be safe
            response = await self._make_request("api/embeddings", data, timeout=120)

            embedding_vector = response.get("embedding", [])
            # Ollama doesn't provide token counts for embeddings
            tokens_used = max(1, len(text_input) // 4)  # Rough estimation
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Ollama is free (local), so cost is 0
            return EmbeddingResult(
                provider_name="ollama",
                model_name=model,
                embedding=embedding_vector,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost_estimate_usd=0.0,
            )

        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")

    async def analyze_image(
        self, image_data: bytes, prompt: str, model: str = "llava-phi3:latest", **kwargs
    ) -> ImageAnalysisResult:
        """Analyze image using Ollama's API (requires vision model like llava)."""
        start_time = datetime.now()

        try:
            # Convert image to base64
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            # Ollama vision models use the chat endpoint with images
            # The images field should be at the message level
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt, "images": [image_base64]}],
                "stream": False,
                **kwargs,
            }

            # Image analysis can take a long time on CPU, use 5 minute timeout
            response = await self._make_request("api/chat", data)

            # Check if response has the expected structure
            if "message" not in response:
                raise Exception(f"Unexpected response format: {response}")

            result = response.get("message", {}).get("content", "")
            if not result:
                raise Exception(f"Empty response from model. Response: {response}")

            tokens_used = response.get("eval_count", 0) + response.get("prompt_eval_count", 0)
            if tokens_used == 0:
                tokens_used = max(1, len(prompt + result) // 4)

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Ollama is free (local), so cost is 0
            return ImageAnalysisResult(
                objects=[],
                captions=[result] if result else [],
                text=result,
                provider_name="ollama",
                model_name=model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost_estimate_usd=0.0,
            )

        except Exception as e:
            # Preserve the original error message
            error_msg = str(e) if e else repr(e)
            if not error_msg or error_msg.strip() == "" or error_msg == "Ollama request error: ":
                error_msg = f"Failed to analyze image with model '{model}'. Make sure the model is installed and supports vision (e.g., 'ollama pull {model}')"
            # If the error already contains "Ollama", just re-raise it, otherwise wrap it
            if "Ollama" in error_msg:
                raise Exception(error_msg)
            raise Exception(f"Ollama API error: {error_msg}")

    def _get_metadata(self) -> ProviderMetadata:
        """Return metadata about the Ollama provider."""
        return ProviderMetadata(
            name="ollama",
            version="1.0.0",
            capabilities=[
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.CHAT,
                ProviderCapability.EMBEDDINGS,
                ProviderCapability.IMAGE_ANALYSIS,
                ProviderCapability.CODE_GENERATION,
            ],
            pricing={
                "llama2": {"input": 0.0, "output": 0.0},
                "mistral": {"input": 0.0, "output": 0.0},
                "llava": {"input": 0.0, "output": 0.0},
                "codellama": {"input": 0.0, "output": 0.0},
            },
            typical_latency_ms={"llama2": 500, "mistral": 400, "llava": 800, "codellama": 600},
            latency={
                "llama2": {"p50": 500, "p95": 1500},
                "mistral": {"p50": 400, "p95": 1200},
                "llava": {"p50": 800, "p95": 2500},
                "codellama": {"p50": 600, "p95": 1800},
            },
            max_context_length=4096,
            max_tokens_per_request=2048,
            supported_models=["llama2", "mistral", "llava", "codellama", "phi", "gemma", "qwen"],
        )

    async def get_cost_estimate(
        self, operation: str, input_tokens: int, output_tokens: Optional[int] = None, **kwargs
    ) -> float:
        """Estimate cost for an operation (Ollama is free)."""
        return 0.0

    async def get_latency_estimate(self, operation: str, **kwargs) -> float:
        """Estimate latency for an operation."""
        model = kwargs.get("model", "llama2")
        if self.metadata.latency:
            latency = self.metadata.latency.get(model, {"p50": 0, "p95": 0})
            return latency["p50"]  # Return median latency
        return 0.0

    async def list_models(self) -> List[str]:
        """List all available models from Ollama."""
        try:
            url = f"{self.base_url}/api/tags"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error ({response.status}): {error_text}")
                    data = await response.json()
                    # Extract model names from the response
                    models = []
                    for model_info in data.get("models", []):
                        model_name = model_info.get("name", "")
                        if model_name:
                            models.append(model_name)
                    return models
        except Exception as e:
            raise Exception(f"Failed to list Ollama models: {str(e)}")
