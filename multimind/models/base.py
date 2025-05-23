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
        self.cost_per_token: Optional[float] = None
        self.avg_latency: Optional[float] = None

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
        pass

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
        """Generate chat completion stream from the model."""
        pass

    @abstractmethod
    async def embeddings(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for the input text."""
        pass

    async def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate the cost of a request based on token usage."""
        if self.cost_per_token is None:
            return 0.0
        return (prompt_tokens + completion_tokens) * self.cost_per_token

    async def get_latency(self) -> Optional[float]:
        """Get the average latency for this model."""
        return self.avg_latency