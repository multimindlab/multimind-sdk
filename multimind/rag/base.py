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

    @abstractmethod
    def evaluate(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the quality of results for a given query."""
        pass

    @abstractmethod
    def parse_output(self, response: str) -> Union[str, Dict[str, Any]]:
        """Parse the output from the LLM into structured data or text."""
        pass

    @abstractmethod
    def validate_output(self, response: str, schema: Optional[Dict[str, Any]] = None) -> bool:
        """Validate the output against a schema."""
        pass

    @abstractmethod
    def format_output(self, response: str, output_format: str = "text") -> Union[str, Dict[str, Any]]:
        """Format the output into the specified format (e.g., JSON, XML, Markdown)."""
        pass