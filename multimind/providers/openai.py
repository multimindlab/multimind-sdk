"""
OpenAI provider adapter for the MultimindSDK.
"""

from typing import Dict, List, Optional, Union, Any
import openai
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

class OpenAIProvider(ProviderAdapter):
    """OpenAI provider adapter implementation."""
    
    def __init__(self, config: ProviderConfig):
        """Initialize the OpenAI provider adapter."""
        super().__init__(config)
        self.client = openai.AsyncOpenAI(api_key=config.api_key)
        self.metadata = ProviderMetadata(
            name="openai",
            version="1.0.0",
            capabilities=[
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.CHAT,
                ProviderCapability.EMBEDDINGS,
                ProviderCapability.IMAGE_ANALYSIS,
                ProviderCapability.CODE_GENERATION
            ],
            pricing={
                "gpt-4": 0.03,
                "gpt-3.5-turbo": 0.0015,
                "text-embedding-ada-002": 0.0001
            },
            typical_latency_ms={
                "gpt-4": 500,
                "gpt-3.5-turbo": 200
            },
            max_context_length=4096,
            max_tokens_per_request=2048,
            supported_models=["gpt-4", "gpt-3.5-turbo", "text-embedding-ada-002"]
        )
    
    async def generate_text(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        **kwargs
    ) -> GenerationResult:
        """Generate text using OpenAI's API."""
        start_time = datetime.now()
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
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
                input_cost * response.usage.prompt_tokens +
                output_cost * response.usage.completion_tokens
            ) / 1000  # Convert to USD
            
            return GenerationResult(
                provider_name="openai",
                model_name=model,
                result=result,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost_estimate_usd=cost
            )
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        **kwargs
    ) -> GenerationResult:
        """Generate chat completion using OpenAI's API."""
        start_time = datetime.now()
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
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
                input_cost * response.usage.prompt_tokens +
                output_cost * response.usage.completion_tokens
            ) / 1000  # Convert to USD
            
            return GenerationResult(
                provider_name="openai",
                model_name=model,
                result=result,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost_estimate_usd=cost
            )
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def generate_embeddings(
        self,
        text: str,
        model: str = "text-embedding-ada-002",
        **kwargs
    ) -> EmbeddingResult:
        """Generate embeddings using OpenAI's API."""
        start_time = datetime.now()
        
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=text,
                **kwargs
            )
            
            embeddings = response.data[0].embedding
            tokens_used = response.usage.total_tokens
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Calculate cost based on model pricing
            pricing = self.metadata.pricing.get(model, {"input": 0.0})
            cost = pricing["input"] * tokens_used / 1000  # Convert to USD
            
            return EmbeddingResult(
                provider_name="openai",
                model_name=model,
                embeddings=embeddings,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost_estimate_usd=cost
            )
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        model: str = "gpt-4-vision-preview",
        **kwargs
    ) -> ImageAnalysisResult:
        """Analyze image using OpenAI's API."""
        start_time = datetime.now()
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data.hex()}"
                                }
                            }
                        ]
                    }
                ],
                **kwargs
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
                input_cost * response.usage.prompt_tokens +
                output_cost * response.usage.completion_tokens
            ) / 1000  # Convert to USD
            
            return ImageAnalysisResult(
                provider_name="openai",
                model_name=model,
                result=result,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost_estimate_usd=cost
            )
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _get_metadata(self) -> ProviderMetadata:
        """Return metadata about the OpenAI provider."""
        return self.metadata

    def get_cost_estimate(self, model: str, tokens: int) -> float:
        """Estimate the cost for a given model and token usage."""
        pricing = self.metadata.pricing.get(model, {"input": 0.0, "output": 0.0})
        return (pricing["input"] + pricing["output"]) * tokens / 1000  # Convert to USD

    def get_latency_estimate(self, model: str) -> Dict[str, int]:
        """Return latency estimates for a given model."""
        return self.metadata.latency.get(model, {"p50": 0, "p95": 0})
    
    async def estimate_cost(
        self,
        task_type: str,
        model: str,
        input_tokens: int,
        output_tokens: Optional[int] = None
    ) -> float:
        """Estimate cost for a given task."""
        pricing = self.metadata.pricing.get(model, {"input": 0.0, "output": 0.0})
        
        if task_type == "embeddings":
            return pricing["input"] * input_tokens / 1000
        else:
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
        latency = self.metadata.latency.get(model, {"p50": 0, "p95": 0})
        return latency["p50"]  # Return median latency