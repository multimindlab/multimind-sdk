"""
Enhanced base class for RAG (Retrieval Augmented Generation) implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import asyncio
from ..models.base import BaseLLM

class RAGError(Exception):
    """Base exception for RAG-related errors."""
    pass

class DocumentProcessingError(RAGError):
    """Raised when there's an error processing documents."""
    pass

class RetrievalError(RAGError):
    """Raised when there's an error during retrieval."""
    pass

class GenerationError(RAGError):
    """Raised when there's an error during generation."""
    pass

@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality."""
    precision: float
    recall: float
    f1_score: float
    relevance_scores: List[float]
    latency_ms: float

@dataclass
class GenerationMetrics:
    """Metrics for generation quality."""
    answer_relevance: float
    faithfulness: float
    hallucination_score: float
    latency_ms: float
    token_usage: Dict[str, int]

class RetrievalStrategy(Enum):
    """Different retrieval strategies available."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    MULTI_VECTOR = "multi_vector"
    CROSS_ENCODER = "cross_encoder"

class ChunkingStrategy(Enum):
    """Different document chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    SLIDING_WINDOW = "sliding_window"

@runtime_checkable
class AsyncVectorStore(Protocol):
    """Protocol for async vector store operations."""
    async def add(self, vectors: List[List[float]], documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        ...
    async def search(self, query_vector: List[float], k: int, **kwargs) -> List[Dict[str, Any]]:
        ...
    async def clear(self) -> None:
        ...

class BaseRAG(ABC):
    """Enhanced abstract base class for RAG implementations."""

    def __init__(
        self,
        embedder: BaseLLM,
        vector_store: AsyncVectorStore,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.DENSE,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
        **kwargs
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.retrieval_strategy = retrieval_strategy
        self.chunking_strategy = chunking_strategy
        self.kwargs = kwargs
        self._semaphore = asyncio.Semaphore(kwargs.get('max_concurrent_operations', 10))

    async def _execute_with_semaphore(self, coro):
        """Execute coroutine with semaphore for rate limiting."""
        async with self._semaphore:
            return await coro

    @abstractmethod
    async def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        chunking_strategy: Optional[ChunkingStrategy] = None,
        **kwargs
    ) -> None:
        """Add documents to the vector store with enhanced error handling. Must be implemented in subclass."""
        raise NotImplementedError("add_documents must be implemented in a subclass of BaseRAG.")

    @abstractmethod
    async def search(
        self,
        query: str,
        k: int = 3,
        retrieval_strategy: Optional[RetrievalStrategy] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents with enhanced retrieval strategies. Must be implemented in subclass."""
        raise NotImplementedError("search must be implemented in a subclass of BaseRAG.")

    @abstractmethod
    async def query(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Query the RAG system with token budget management.
        
        Args:
            query: Query string
            context: Optional pre-fetched context
            max_tokens: Optional token budget
            **kwargs: Additional generation parameters
        """
        pass

    @abstractmethod
    async def evaluate_retrieval(
        self,
        query: str,
        results: List[Dict[str, Any]],
        ground_truth: Optional[List[Dict[str, Any]]] = None
    ) -> RetrievalMetrics:
        """Evaluate retrieval quality with comprehensive metrics."""
        pass

    @abstractmethod
    async def evaluate_generation(
        self,
        query: str,
        response: str,
        context: List[Dict[str, Any]],
        ground_truth: Optional[str] = None
    ) -> GenerationMetrics:
        """Evaluate generation quality with comprehensive metrics."""
        pass

    @abstractmethod
    async def optimize_context(
        self,
        query: str,
        context: List[Dict[str, Any]],
        max_tokens: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Optimize context based on relevance and token budget."""
        pass

    @abstractmethod
    async def generate_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]],
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """Generate optimized prompt with optional few-shot examples."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear the vector store with proper cleanup."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics and health metrics."""
        pass

    @abstractmethod
    async def validate_documents(
        self,
        documents: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Validate documents before processing."""
        pass

    @abstractmethod
    async def reindex(
        self,
        documents: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """Reindex documents with optional filtering."""
        pass