"""
Core provider interface and structures for the MultimindSDK.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from pydantic import BaseModel
from datetime import datetime

class ProviderCapability(str, Enum):
    """Capabilities that a provider can support."""
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    IMAGE_ANALYSIS = "image_analysis"
    CODE_GENERATION = "code_generation"
    FINE_TUNING = "fine_tuning"

class ProviderConfig(BaseModel):
    """Base configuration for a provider."""
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_retries: int = 3
    timeout: int = 30
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class ProviderMetadata(BaseModel):
    """Metadata about a provider's capabilities and limits."""
    name: str
    version: str
    capabilities: List[ProviderCapability]
    max_context_length: int
    max_tokens_per_request: int
    pricing: Dict[str, float]  # e.g. {"input": 0.001, "output": 0.002}
    typical_latency_ms: Dict[str, int]  # e.g. {"text_generation": 200}
    supported_models: List[str]

class GenerationResult(BaseModel):
    """Standardized result from text generation."""
    text: str
    tokens_used: int
    provider_name: str
    model_name: str
    latency_ms: float
    cost_estimate_usd: float
    metadata: Dict[str, Any] = {}
    created_at: datetime = datetime.now()

class EmbeddingResult(BaseModel):
    """Standardized result from embeddings generation."""
    embedding: List[float]
    tokens_used: int
    provider_name: str
    model_name: str
    latency_ms: float
    cost_estimate_usd: float
    metadata: Dict[str, Any] = {}

class ImageAnalysisResult(BaseModel):
    """Standardized result from image analysis."""
    objects: List[Dict[str, Any]]
    captions: List[str]
    text: Optional[str]  # OCR text if any
    provider_name: str
    model_name: str
    latency_ms: float
    cost_estimate_usd: float
    metadata: Dict[str, Any] = {}

class ProviderAdapter(ABC):
    """Base class for provider adapters."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.metadata = self._get_metadata()
    
    @abstractmethod
    def _get_metadata(self) -> ProviderMetadata:
        """Get provider metadata including capabilities and limits."""
        pass
    
    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        model: str,
        **kwargs
    ) -> GenerationResult:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> GenerationResult:
        """Generate chat completion."""
        pass
    
    @abstractmethod
    async def generate_embeddings(
        self,
        text: Union[str, List[str]],
        model: str,
        **kwargs
    ) -> EmbeddingResult:
        """Generate embeddings for text."""
        pass
    
    @abstractmethod
    async def analyze_image(
        self,
        image_data: bytes,
        model: str,
        **kwargs
    ) -> ImageAnalysisResult:
        """Analyze an image."""
        pass
    
    @abstractmethod
    async def get_cost_estimate(
        self,
        operation: str,
        input_tokens: int,
        output_tokens: Optional[int] = None,
        **kwargs
    ) -> float:
        """Estimate cost for an operation."""
        pass
    
    @abstractmethod
    async def get_latency_estimate(
        self,
        operation: str,
        **kwargs
    ) -> float:
        """Estimate latency for an operation."""
        pass 