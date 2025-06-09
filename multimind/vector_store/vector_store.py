"""
Main vector store implementation that manages different backends.
"""

import logging
from typing import List, Dict, Any, Optional

from .base import VectorStoreBackend, VectorStoreConfig, SearchResult, VectorStoreType
from .backends.faiss import FAISSBackend
from .chroma import ChromaBackend
from .backends.weaviate import WeaviateBackend
from .backends.qdrant import QdrantBackend
from .backends.milvus import MilvusBackend
from .backends.pinecone import PineconeBackend
from .backends.elasticsearch import ElasticsearchBackend
from .backends.redis import RedisBackend
from .backends.postgres import PostgresBackend

class VectorStore:
    """Unified vector store interface."""
    
    def __init__(self, config: VectorStoreConfig):
        """
        Initialize vector store.
        
        Args:
            config: Vector store configuration
        """
        self.config = config
        self.backend = self._get_backend()
        self.logger = logging.getLogger(__name__)

    def _get_backend(self) -> VectorStoreBackend:
        """Get appropriate vector store backend."""
        store_type = VectorStoreType(self.config.store_type)
        
        backend_map = {
            VectorStoreType.FAISS: FAISSBackend,
            VectorStoreType.CHROMA: ChromaBackend,
            VectorStoreType.WEAVIATE: WeaviateBackend,
            VectorStoreType.QDRANT: QdrantBackend,
            VectorStoreType.MILVUS: MilvusBackend,
            VectorStoreType.PINECONE: PineconeBackend,
            VectorStoreType.ELASTICSEARCH: ElasticsearchBackend,
            VectorStoreType.REDIS: RedisBackend,
            VectorStoreType.POSTGRES: PostgresBackend
        }
        
        backend_class = backend_map.get(store_type)
        if not backend_class:
            raise ValueError(f"Unsupported vector store type: {store_type}")
        
        return backend_class(self.config)

    async def initialize(self) -> None:
        """Initialize vector store backend."""
        await self.backend.initialize()

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to store."""
        await self.backend.add_vectors(vectors, metadatas, documents, ids)

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search vectors in store."""
        return await self.backend.search(query_vector, k, filter_criteria)

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from store."""
        await self.backend.delete_vectors(ids)

    async def clear(self) -> None:
        """Clear vector store."""
        await self.backend.clear()

    async def persist(self, path: str) -> None:
        """Persist vector store to disk."""
        await self.backend.persist(path)

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "VectorStore":
        """Load vector store from disk."""
        store = cls(config)
        store.backend = await store.backend.load(path, config)
        return store 