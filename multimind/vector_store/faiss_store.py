"""
Deprecated alias module for the FAISS backend.

The canonical implementation lives in multimind.vector_store.faiss
(FAISSBackend). The FAISSVectorStore class that used to live here was
broken (it imported a non-existent base class) and has been replaced by
an alias to FAISSBackend. Import FAISSBackend directly instead.
"""

from .faiss import FAISSBackend

# Deprecated: kept so existing imports keep working
FAISSVectorStore = FAISSBackend

__all__ = ["FAISSBackend", "FAISSVectorStore"]
