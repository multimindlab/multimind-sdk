"""
Claude provider adapter for the MultimindSDK.
"""

from typing import Dict, List, Optional, Union, Any
import anthropic
import logging
from datetime import datetime
from ..core.provider import (
    ProviderAdapter,
    ProviderConfig,
    ProviderMetadata,
    ProviderCapability,
    GenerationResult,
    EmbeddingResult,
    ImageAnalysisResult
)

logger = logging.getLogger(__name__)

class ClaudeProvider(ProviderAdapter):
    """Claude provider adapter implementation."""
    
    def __init__(self, config: ProviderConfig):
        """Initialize the Claude provider adapter."""
        super().__init__(config)
        self.client = anthropic.AsyncAnthropic(api_key=config.api_key)
    
    async def generate_text(
        self,
        model: str,
        prompt: str,
        **kwargs
    ) -> GenerationResult:
        """Generate text using Claude's API."""
        start_time = datetime.now()
        
        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=kwargs.pop("max_tokens", 1000),
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            result = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Calculate cost based on model pricing
            pricing = self.metadata.pricing.get(model, {"input": 0.0, "output": 0.0})
            cost = (
                pricing["input"] * response.usage.input_tokens +
                pricing["output"] * response.usage.output_tokens
            ) / 1000  # Convert to USD
            return GenerationResult(
                text=result,
                tokens_used=tokens_used,
                provider_name="claude",
                model_name=model,
                latency_ms=latency_ms,
                cost_estimate_usd=cost
            )
        except AttributeError:
            logger.error("The Claude API client is missing the 'messages.create' method. Please update the client.")
            raise RuntimeError("Claude API client is outdated or incompatible.")
        except Exception as e:
            logger.error(f"Error generating text with Claude API: {e}")
            raise
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-3-sonnet",
        **kwargs
    ) -> GenerationResult:
        """Generate chat completion using Claude's API."""
        start_time = datetime.now()
        
        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=kwargs.pop("max_tokens", 1000),
                messages=messages,
                **kwargs
            )
            
            result = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Calculate cost based on model pricing
            pricing = self.metadata.pricing.get(model, {"input": 0.0, "output": 0.0})
            cost = (
                pricing["input"] * response.usage.input_tokens +
                pricing["output"] * response.usage.output_tokens
            ) / 1000  # Convert to USD
            
            return GenerationResult(
                provider_name="claude",
                model_name=model,
                result=result,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost_estimate_usd=cost
            )
            
        except Exception as e:
            raise Exception(f"Claude API error: {str(e)}")
    
    async def generate_embeddings(
        self,
        text: str,
        model: str = "claude-3-sonnet",
        **kwargs
    ) -> EmbeddingResult:
        """Generate embeddings using Claude's API."""
        raise NotImplementedError("Claude does not support embeddings")
    
    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        model: str = "claude-3-sonnet",
        **kwargs
    ) -> ImageAnalysisResult:
        """Analyze image using Claude's API."""
        start_time = datetime.now()
        
        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=kwargs.pop("max_tokens", 1000),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data.hex()
                                }
                            }
                        ]
                    }
                ],
                **kwargs
            )
            
            result = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Calculate cost based on model pricing
            pricing = self.metadata.pricing.get(model, {"input": 0.0, "output": 0.0})
            cost = (
                pricing["input"] * response.usage.input_tokens +
                pricing["output"] * response.usage.output_tokens
            ) / 1000  # Convert to USD
            
            return ImageAnalysisResult(
                provider_name="claude",
                model_name=model,
                result=result,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost_estimate_usd=cost
            )
            
        except Exception as e:
            raise Exception(f"Claude API error: {str(e)}")
    
    async def estimate_cost(
        self,
        task_type: str,
        model: str,
        input_tokens: int,
        output_tokens: Optional[int] = None
    ) -> float:
        """Estimate cost for a given task."""
        pricing = self.metadata.pricing.get(model, {"input": 0.0, "output": 0.0})
        return (
            pricing["input"] * input_tokens +
            pricing["output"] * (output_tokens or 0)
        ) / 1000
    
    async def estimate_latency(
        self,
        task_type: str,
        model: str,
        input_tokens: int,
        output_tokens: Optional[int] = None
    ) -> float:
        """Estimate latency for a given task."""
        latency = self.metadata.latency.get(model, {"p50": 0, "p95": 0}) if self.metadata.latency else {"p50": 0, "p95": 0}
        return latency["p50"]  # Return median latency
    
    async def get_cost_estimate(
        self,
        operation: str,
        input_tokens: int,
        output_tokens: Optional[int] = None,
        **kwargs
    ) -> float:
        """Estimate cost for an operation (abstract method implementation)."""
        # Extract model from kwargs or use default
        model = kwargs.get("model", "claude-3-sonnet")
        pricing = self.metadata.pricing.get(model, {"input": 0.0, "output": 0.0})
        
        if operation == "embeddings":
            return pricing["input"] * input_tokens / 1000
        else:
            return (
                pricing["input"] * input_tokens +
                pricing["output"] * (output_tokens or 0)
            ) / 1000
    
    async def get_latency_estimate(
        self,
        operation: str,
        **kwargs
    ) -> float:
        """Estimate latency for an operation (abstract method implementation)."""
        # Extract model from kwargs or use default
        model = kwargs.get("model", "claude-3-sonnet")
        if self.metadata.latency:
            latency = self.metadata.latency.get(model, {"p50": 0, "p95": 0})
            return latency["p50"]  # Return median latency
        return 0.0
    
    def _get_metadata(self) -> ProviderMetadata:
        """Return metadata about the Claude provider."""
        return ProviderMetadata(
            name="claude",
            version="1.0.0",
            capabilities=[
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.CHAT,
                ProviderCapability.CODE_GENERATION
            ],
            pricing={
                "claude-3-opus": {"input": 0.015, "output": 0.075},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
            },
            typical_latency_ms={
                "claude-3-opus": 800,
                "claude-3-sonnet": 400,
                "claude-3-haiku": 200
            },
            latency={
                "claude-3-opus": {"p50": 800, "p95": 3000},
                "claude-3-sonnet": {"p50": 400, "p95": 1500},
                "claude-3-haiku": {"p50": 200, "p95": 800}
            },
            max_context_length=200000,
            max_tokens_per_request=4096,
            supported_models=["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        )