"""
Base classes and interfaces for vector store implementations.
"""

from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum

@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    store_type: str  # Type of vector store to use
    dimension: int  # Vector dimension
    index_params: Dict[str, Any]  # Index-specific parameters
    connection_params: Dict[str, Any]  # Connection parameters
    search_params: Dict[str, Any]  # Search parameters
    custom_params: Dict[str, Any]  # Custom parameters

@dataclass
class SearchResult:
    """Represents a search result."""
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    document: Dict[str, Any]
    score: float

class VectorStoreType(Enum):
    """Types of vector stores supported."""
    FAISS = "faiss"
    CHROMA = "chroma"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    PINECONE = "pinecone"
    ELASTICSEARCH = "elasticsearch"
    REDIS = "redis"
    POSTGRES = "postgres"

@runtime_checkable
class VectorStoreBackend(Protocol):
    """Protocol defining vector store backend interface."""
    async def initialize(self) -> None:
        """Initialize the vector store."""
        pass

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to the store."""
        pass

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        pass

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from the store."""
        pass

    async def clear(self) -> None:
        """Clear all vectors from the store."""
        pass

    async def persist(self, path: str) -> None:
        """Persist the vector store to disk."""
        pass

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "VectorStoreBackend":
        """Load vector store from disk."""
        pass """
Base vector store interface and embedding standardization.
"""

from typing import Dict, List, Optional, Any, Union, Protocol
from pydantic import BaseModel
import numpy as np
from abc import ABC, abstractmethod

class EmbeddingDimension(str, Enum):
    """Standard embedding dimensions for different providers."""
    OPENAI_ADA = "1536"
    OPENAI_LARGE = "3072"
    CLAUDE = "4096"
    COHERE = "1024"
    HUGGINGFACE = "768"

class EmbeddingMetadata(BaseModel):
    """Metadata for an embedding."""
    provider: str
    model: str
    dimension: int
    metadata: Dict[str, Any] = {}

class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    dimension: int
    similarity_metric: str = "cosine"  # cosine, euclidean, dot
    index_type: str = "flat"  # flat, ivf, hnsw
    metadata: Dict[str, Any] = {}

class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    async def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        **kwargs
    ) -> List[str]:
        """Add vectors to the store."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete_vectors(
        self,
        vector_ids: List[str],
        **kwargs
    ) -> bool:
        """Delete vectors from the store."""
        pass
    
    @abstractmethod
    async def get_vector(
        self,
        vector_id: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Get a vector by ID."""
        pass
    
    @abstractmethod
    async def update_metadata(
        self,
        vector_id: str,
        metadata: Dict[str, Any],
        **kwargs
    ) -> bool:
        """Update metadata for a vector."""
        pass

class EmbeddingStandardizer:
    """Standardizes embeddings from different providers."""
    
    def __init__(self, target_dimension: int = 1536):
        """Initialize the standardizer."""
        self.target_dimension = target_dimension
    
    def standardize(
        self,
        embedding: np.ndarray,
        source_dimension: int,
        **kwargs
    ) -> np.ndarray:
        """Standardize an embedding to the target dimension."""
        if embedding.shape[0] == self.target_dimension:
            return embedding
        
        # If source dimension is larger, use PCA
        if source_dimension > self.target_dimension:
            return self._reduce_dimension(embedding)
        
        # If source dimension is smaller, use zero padding
        return self._pad_dimension(embedding)
    
    def _reduce_dimension(self, embedding: np.ndarray) -> np.ndarray:
        """Reduce embedding dimension using PCA."""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.target_dimension)
        return pca.fit_transform(embedding.reshape(1, -1)).flatten()
    
    def _pad_dimension(self, embedding: np.ndarray) -> np.ndarray:
        """Pad embedding to target dimension with zeros."""
        padded = np.zeros(self.target_dimension)
        padded[:embedding.shape[0]] = embedding
        return padded

class VectorStoreFactory:
    """Factory for creating vector stores."""
    
    @staticmethod
    def create_store(
        store_type: str,
        config: VectorStoreConfig,
        **kwargs
    ) -> VectorStore:
        """Create a vector store instance."""
        if store_type == "faiss":
            from .faiss_store import FAISSVectorStore
            return FAISSVectorStore(config)
        elif store_type == "milvus":
            from .milvus_store import MilvusVectorStore
            return MilvusVectorStore(config)
        elif store_type == "pinecone":
            from .pinecone_store import PineconeVectorStore
            return PineconeVectorStore(config)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}") 