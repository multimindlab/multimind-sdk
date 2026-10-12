"""RAG (Retrieval Augmented Generation) module.

Requires the ``rag`` extras: ``pip install 'multimind-sdk[rag]'``.
"""

try:
    from .base import BaseRAG, RAGError
    from .graph_retrieval import GraphRetriever
    from .postprocessing import PostProcessingConfig, PostProcessor
    from .rag import RAG, RAGConfig
except ImportError as exc:  # pragma: no cover - exercised on minimal installs
    raise ImportError(
        "RAG features require additional dependencies. "
        "Install with: pip install 'multimind-sdk[rag]'"
    ) from exc

__all__ = [
    "RAG",
    "RAGConfig",
    "BaseRAG",
    "RAGError",
    "GraphRetriever",
    "PostProcessor",
    "PostProcessingConfig",
]
