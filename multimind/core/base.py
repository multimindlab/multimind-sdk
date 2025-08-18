"""
Base class for all LLM implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncGenerator

class BaseLLM(ABC):
    """Abstract base class for all LLM implementations."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from the model."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text stream from the model."""
        yield ""  # Placeholder to make it an async generator

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate chat completion from the model."""
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Chat stream with the model."""
        yield ""  # Placeholder to make it an async generator

    @abstractmethod
    async def embeddings(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for the input text."""
        pass

    async def get_quality(self) -> Optional[float]:
        """Get the quality score for this model."""
        return None  # Placeholder implementation
    
    async def get_cost(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> float:
        """Get the cost estimate for this model."""
        return 0.0  # Placeholder implementation
    
    async def get_latency(self) -> Optional[float]:
        """Get the latency estimate for this model."""
        return None  # Placeholder implementation