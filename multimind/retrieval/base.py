"""
Base classes and interfaces for retrieval implementations.
"""

from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum

@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    retriever_type: str  # Type of retriever to use
    vector_store_config: Dict[str, Any]  # Vector store configuration
    search_params: Dict[str, Any]  # Search parameters
    custom_params: Dict[str, Any]  # Custom parameters

@dataclass
class RetrievalResult:
    """Represents a retrieval result."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str

class RetrieverType(Enum):
    """Types of retrievers supported."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"

@runtime_checkable
class Retriever(Protocol):
    """Protocol defining retriever interface."""
    async def initialize(self) -> None:
        """Initialize the retriever."""
        pass

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents."""
        pass

    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to the retriever."""
        pass

    async def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from the retriever."""
        pass

    async def clear(self) -> None:
        """Clear all documents from the retriever."""
        pass 