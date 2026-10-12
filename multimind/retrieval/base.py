"""
Base classes and interfaces for retrieval implementations.

DEPRECATION / DUPLICATION NOTE (module consolidation review): this module is
not imported by ``multimind.retrieval.__init__`` and is not referenced
anywhere else in the codebase. The package's public surface (``Retriever``,
``RetrievalConfig``, ``RetrievalResult``) comes from ``retriever.py``
instead. Every class here has a same-named counterpart elsewhere in the
package that is NOT a drop-in match, so nothing below is aliased:

- ``RetrievalConfig`` here (``retriever_type``/``vector_store_config``/
  ``search_params``/``custom_params``) differs from the canonical
  ``retriever.RetrievalConfig`` (``vector_store``/``document_processor``/
  ``embedding_generator``/``top_k``/``similarity_threshold``).
- ``RetrievalResult`` here (``id``/``content``/``metadata``/``score``/
  ``source``) is a *third* distinct shape alongside ``retriever.RetrievalResult``
  (the canonical one, exported in ``__all__``) and ``retrieval.RetrievalResult``
  (``document``/``metadata``/``score``/``retrieval_type``/``reranking_score``,
  used internally by ``HybridRetriever``).
- ``RetrieverType`` (``DENSE``/``SPARSE``/``HYBRID``) is a different concept
  from ``retrieval.QueryType`` and ``enhanced_retrieval.RetrievalType``.
- ``Retriever`` here is a structural ``Protocol`` with a different
  ``retrieve()`` signature (``k``, ``filter_criteria``) than the canonical
  concrete ``retriever.Retriever`` class (``top_k``, ``**kwargs``).

Kept only for any external code that may import
``multimind.retrieval.base`` directly.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


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
        self, query: str, k: int = 5, filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents."""
        pass

    async def add_documents(
        self, documents: List[Dict[str, Any]], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to the retriever."""
        pass

    async def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from the retriever."""
        pass

    async def clear(self) -> None:
        """Clear all documents from the retriever."""
        pass
