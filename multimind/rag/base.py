"""
Base class for RAG (Retrieval Augmented Generation) implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from ..models.base import BaseLLM

class BaseRAG(ABC):
    """Abstract base class for RAG implementations."""
    
    def __init__(
        self,
        embedder: BaseLLM,
        vector_store: Any,  # Type depends on implementation
        **kwargs
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.kwargs = kwargs
        
    @abstractmethod
    async def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """Add documents to the vector store."""
        pass
        
    @abstractmethod
    async def search(
        self,
        query: str,
        k: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        pass
        
    @abstractmethod
    async def query(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> str:
        """Query the RAG system with optional context."""
        pass
        
    @abstractmethod
    async def clear(self) -> None:
        """Clear the vector store."""
        pass 