"""
Base classes and interfaces for embedding generation.
"""

from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str  # Name of the embedding model
    dimension: int  # Dimension of the embeddings
    batch_size: int = 32  # Batch size for generation
    device: str = "cpu"  # Device to use for generation
    custom_params: Dict[str, Any] = None  # Custom parameters

class EmbeddingType(Enum):
    """Types of embedding models supported."""
    OPENAI = "openai"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    HUGGINGFACE = "huggingface"

@runtime_checkable
class EmbeddingGenerator(Protocol):
    """Protocol defining embedding generator interface."""
    async def initialize(self) -> None:
        """Initialize the embedding generator."""
        pass

    async def generate(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass

    async def generate_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @property
    def dimension(self) -> int:
        """Get the dimension of the embeddings."""
        pass 