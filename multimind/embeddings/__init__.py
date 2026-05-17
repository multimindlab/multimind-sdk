"""Embeddings module for text embedding generation.

Requires the ``rag`` extras (``sentence-transformers``, ``numpy``, …):
``pip install 'multimind-sdk[rag]'``.
"""

try:
    from .embeddings import EmbeddingGenerator, EmbeddingConfig
    from .embedding import Embedding, EmbeddingType
    from .standardizer import EmbeddingStandardizer
except ImportError as exc:  # pragma: no cover - exercised on minimal installs
    raise ImportError(
        "Embedding features require additional dependencies. "
        "Install with: pip install 'multimind-sdk[rag]'"
    ) from exc

__all__ = [
    'EmbeddingGenerator',
    'EmbeddingConfig',
    'Embedding',
    'EmbeddingType',
    'EmbeddingStandardizer'
] 