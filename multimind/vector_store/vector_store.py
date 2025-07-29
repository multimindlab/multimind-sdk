"""
Main vector store implementation that manages different backends.
this file that provides the unified abstraction for all vector store backends.
It allows users to switch between databases seamlessly by specifying the backend 
in the config, without changing their code.
All backend implementations (e.g., Milvus, Pinecone, Qdrant, etc.) are mapped in 
this file, so the user can select any supported backend via configuration.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Type

from .base import VectorStoreBackend, VectorStoreConfig, SearchResult, VectorStoreType

def _load_backend(backend_type: VectorStoreType) -> Optional[Type[VectorStoreBackend]]:
    """Lazily load a backend only when requested."""
    # Map backend types to their module paths
    backend_modules = {
        VectorStoreType.FAISS: '.faiss',
        VectorStoreType.CHROMA: '.chroma',
        VectorStoreType.WEAVIATE: '.weaviate',
        VectorStoreType.QDRANT: '.qdrant',
        VectorStoreType.MILVUS: '.milvus',
        VectorStoreType.PINECONE: '.pinecone',
        VectorStoreType.ELASTICSEARCH: '.elasticsearch',
        VectorStoreType.ALIBABACLOUD_OPENSEARCH: '.alibabacloud_opensearch',
        VectorStoreType.ATLAS: '.atlas',
        VectorStoreType.AWADB: '.awadb',
        VectorStoreType.AZURESEARCH: '.azuresearch',
        VectorStoreType.BAGELDB: '.bageldb',
        VectorStoreType.BAIDUCLOUD_VECTOR_SEARCH: '.baiducloud_vector_search',
        VectorStoreType.CASSANDRA: '.cassandra',
        VectorStoreType.CLARIFAI: '.clarifai',
        VectorStoreType.CLICKHOUSE: '.clickhouse',
        VectorStoreType.DATABRICKS_VECTOR_SEARCH: '.databricks_vector_search',
        VectorStoreType.DASHVECTOR: '.dashvector',
        VectorStoreType.DINGO: '.dingo',
        VectorStoreType.ELASTIC_VECTOR_SEARCH: '.elastic_vector_search',
        VectorStoreType.HOLOGRES: '.hologres',
        VectorStoreType.LANCEDB: '.lancedb',
        VectorStoreType.MARQO: '.marqo',
        VectorStoreType.MEILISEARCH: '.meilisearch',
        VectorStoreType.MONGODB_ATLAS: '.mongodb_atlas',
        VectorStoreType.MOMENTO_VECTOR_INDEX: '.momento_vector_index',
        VectorStoreType.NEO4J_VECTOR: '.neo4j_vector',
        VectorStoreType.OPENSEARCH_VECTOR_SEARCH: '.opensearch_vector_search',
        VectorStoreType.PGVECTOR: '.pgvector',
        VectorStoreType.PGVECTO_RS: '.pgvecto_rs',
        VectorStoreType.PGEMBEDDING: '.pgembedding',
        VectorStoreType.NUCLIADB: '.nucliadb',
        VectorStoreType.MYSCALE: '.myscale',
        VectorStoreType.MATCHING_ENGINE: '.matching_engine',
        VectorStoreType.LLM_RAILS: '.llm_rails',
        VectorStoreType.HIPPO: '.hippo',
        VectorStoreType.EPSILLA: '.epsilla',
        VectorStoreType.DEEPLAKE: '.deeplake',
        VectorStoreType.AZURE_COSMOS_DB: '.azure_cosmos_db',
        VectorStoreType.ANNOY: '.annoy',
        VectorStoreType.ASTRADB: '.astradb',
        VectorStoreType.ANALYTICDB: '.analyticdb',
        VectorStoreType.SKLEARN: '.sklearn',
        VectorStoreType.SINGLESTOREDB: '.singlestoredb',
        VectorStoreType.ROCKSETDB: '.rocksetdb',
        VectorStoreType.SQLITEVSS: '.sqlitevss',
        VectorStoreType.STARROCKS: '.starrocks',
        VectorStoreType.SUPABASE: '.supabase',
        VectorStoreType.TAIR: '.tair',
        VectorStoreType.TIGRIS: '.tigris',
        VectorStoreType.TILEDB: '.tiledb',
        VectorStoreType.TIMESCALEVECTOR: '.timescalevector',
        VectorStoreType.TENCENTVECTORDB: '.tencentvectordb',
        VectorStoreType.USEARCH: '.usearch',
        VectorStoreType.VALD: '.vald',
        VectorStoreType.VECTARA: '.vectara',
        VectorStoreType.TYPESENSE: '.typesense',
        VectorStoreType.XATA: '.xata',
        VectorStoreType.ZEP: '.zep',
        VectorStoreType.ZILLIZ: '.zilliz',
    }
    
    if backend_type not in backend_modules:
        return None
    
    module_path = backend_modules[backend_type]
    backend_name = backend_type.value
    
    try:
        # Import the module dynamically
        module = __import__(f'multimind.vector_store{module_path}', fromlist=[backend_name])
        backend_class = getattr(module, backend_name)
        logging.debug(f"✅ {backend_name} loaded successfully on demand")
        return backend_class
    except (ImportError, AttributeError, Exception) as e:
        # Only log if warnings are enabled
        show_warnings = os.getenv('MULTIMIND_SHOW_BACKEND_WARNINGS', 'false').lower() == 'true'
        if show_warnings:
            logging.warning(f"{backend_name} backend not available - {str(e)}")
        else:
            logging.debug(f"{backend_name} backend not available - {str(e)}")
        return None

# Backend registry for lazy loading
_backend_registry: Dict[VectorStoreType, Type[VectorStoreBackend]] = {}

def get_backend_class(backend_type: VectorStoreType) -> Optional[Type[VectorStoreBackend]]:
    """Get a backend class by type, loading it if necessary."""
    if backend_type in _backend_registry:
        return _backend_registry[backend_type]
    
    backend_class = _load_backend(backend_type)
    if backend_class:
        _backend_registry[backend_type] = backend_class
    return backend_class

class VectorStore:
    """
    Unified vector store interface that supports multiple backends.
    """
    
    def __init__(self, config: VectorStoreConfig):
        """
        Initialize the vector store with the specified configuration.
        
        Args:
            config: Configuration for the vector store backend
        """
        self.config = config
        self.backend = None
        self._backend_instance = None

    def _get_backend(self) -> VectorStoreBackend:
        """Get or create the backend instance."""
        if self._backend_instance is None:
            backend_class = get_backend_class(self.config.backend_type)
            if backend_class is None:
                raise ValueError(f"Backend {self.config.backend_type} is not available")
            
            self._backend_instance = backend_class(self.config)
        return self._backend_instance

    async def initialize(self) -> None:
        """Initialize the vector store."""
        backend = self._get_backend()
        await backend.initialize()

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to the store."""
        backend = self._get_backend()
        await backend.add_vectors(vectors, metadatas, documents, ids)

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        backend = self._get_backend()
        return await backend.search(query_vector, k, filter_criteria)

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by their IDs."""
        backend = self._get_backend()
        await backend.delete_vectors(ids)

    async def clear(self) -> None:
        """Clear all vectors from the store."""
        backend = self._get_backend()
        await backend.clear()

    async def persist(self, path: str) -> None:
        """Persist the vector store to disk."""
        backend = self._get_backend()
        await backend.persist(path)

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "VectorStore":
        """Load a vector store from disk."""
        instance = cls(config)
        backend = instance._get_backend()
        await backend.load(path)
        return instance 