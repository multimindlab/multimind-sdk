"""
Vector store package for managing vector storage and retrieval.
"""

import logging
import os
from typing import Dict, Optional, Type

from .base import (
    SearchResult,
    VectorStoreBackend,
    VectorStoreConfig,
    VectorStoreFactory,
    VectorStoreType,
)
from .vector_store import VectorStore

# Configure logging
logger = logging.getLogger(__name__)

_EXPERIMENTAL_BACKENDS = frozenset(
    {
        "AwaDBBackend",
        "BagelDBBackend",
        "BaiduCloudVectorSearchBackend",
        "ClarifaiBackend",
        "ClickHouseBackend",
        "DashVectorBackend",
        "DatabricksVectorSearchBackend",
        "DeepLakeBackend",
        "DingoDBBackend",
        "ElasticVectorSearchBackend",
        "EpsillaBackend",
    }
)

# Backend registry for lazy loading
_backend_registry: Dict[str, Type[VectorStoreBackend]] = {}

# Map backend names to their module paths
_BACKEND_MODULES = {
    "FAISSBackend": ".faiss",
    "ChromaBackend": ".chroma",
    "WeaviateVectorStore": ".weaviate",
    "QdrantBackend": ".qdrant",
    "MilvusBackend": ".milvus",
    "PineconeBackend": ".pinecone",
    "ElasticsearchBackend": ".elasticsearch",
    "AlibabaCloudOpenSearchBackend": ".alibabacloud_opensearch",
    "AtlasBackend": ".atlas",
    "AzureSearchBackend": ".azuresearch",
    "CassandraBackend": ".cassandra",
    "HologresBackend": ".hologres",
    "LanceDBBackend": ".lancedb",
    "MarqoBackend": ".marqo",
    "MeiliSearchBackend": ".meilisearch",
    "MongoDBAtlasBackend": ".mongodb_atlas",
    "MomentoVectorIndexBackend": ".momento_vector_index",
    "Neo4jVectorBackend": ".neo4j_vector",
    "OpenSearchVectorBackend": ".opensearch_vector_search",
    "PGVectorBackend": ".pgvector",
    "PGVectoRSBackend": ".pgvecto_rs",
    "PGEmbeddingBackend": ".pgembedding",
    "NucliaDBBackend": ".nucliadb",
    "MyScaleBackend": ".myscale",
    "MatchingEngineBackend": ".matching_engine",
    "LLMRailsBackend": ".llm_rails",
    "HippoBackend": ".hippo",
    "AzureCosmosDBBackend": ".azure_cosmos_db",
    "AnnoyBackend": ".annoy",
    "AstraDBBackend": ".astradb",
    "AnalyticDBBackend": ".analyticdb",
    "SklearnBackend": ".sklearn",
    "SingleStoreDBBackend": ".singlestoredb",
    "RocksetDBBackend": ".rocksetdb",
    "SQLiteVSSBackend": ".sqlitevss",
    "StarRocksBackend": ".starrocks",
    "SupabaseVectorStore": ".supabase",
    "TairVectorStore": ".tair",
    "TigrisVectorStore": ".tigris",
    "TileDBVectorStore": ".tiledb",
    "TimescaleVectorStore": ".timescalevector",
    "TencentVectorDBVectorStore": ".tencentvectordb",
    "USearchVectorStore": ".usearch",
    "ValdVectorStore": ".vald",
    "VectaraVectorStore": ".vectara",
    "TypesenseVectorStore": ".typesense",
    "XataVectorStore": ".xata",
    "ZepVectorStore": ".zep",
    "ZillizVectorStore": ".zilliz",
}

# Unimplemented backends. Hidden unless MULTIMIND_EXPERIMENTAL_VECTOR_STORES=1.
_EXPERIMENTAL_MODULES = {
    "AwaDBBackend": ".awadb",
    "BagelDBBackend": ".bageldb",
    "BaiduCloudVectorSearchBackend": ".baiducloud_vector_search",
    "ClarifaiBackend": ".clarifai",
    "ClickHouseBackend": ".clickhouse",
    "DashVectorBackend": ".dashvector",
    "DatabricksVectorSearchBackend": ".databricks_vector_search",
    "DeepLakeBackend": ".deeplake",
    "DingoDBBackend": ".dingo",
    "ElasticVectorSearchBackend": ".elastic_vector_search",
    "EpsillaBackend": ".epsilla",
}


def _load_backend(backend_name: str) -> Optional[Type[VectorStoreBackend]]:
    """Lazily load a backend only when requested."""
    if backend_name in _backend_registry:
        return _backend_registry[backend_name]

    backend_modules = _BACKEND_MODULES
    if backend_name in _EXPERIMENTAL_MODULES:
        enabled = os.getenv("MULTIMIND_EXPERIMENTAL_VECTOR_STORES", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if not enabled:
            logger.debug(
                "%s is an unimplemented backend and is not on the default path", backend_name
            )
            return None
        backend_modules = _EXPERIMENTAL_MODULES

    if backend_name not in backend_modules:
        return None

    module_path = backend_modules[backend_name]

    try:
        # Import the module dynamically
        module = __import__(f"multimind.vector_store{module_path}", fromlist=[backend_name])
        backend_class = getattr(module, backend_name)
        _backend_registry[backend_name] = backend_class
        logger.debug(f"✅ {backend_name} loaded successfully on demand")
        return backend_class
    except (ImportError, AttributeError, Exception) as e:
        # Only log if warnings are enabled
        show_warnings = os.getenv("MULTIMIND_SHOW_BACKEND_WARNINGS", "false").lower() == "true"
        if show_warnings:
            logger.warning(f"{backend_name} backend not available - {str(e)}")
        else:
            logger.debug(f"{backend_name} backend not available - {str(e)}")
        return None


def get_available_backends() -> list:
    """Get list of available vector store backends."""
    # This will only return backends that have been loaded
    return list(_backend_registry.keys())


def is_backend_available(backend_name: str) -> bool:
    """Check if a specific backend is available."""
    if backend_name in _backend_registry:
        return True
    # Try to load it
    return _load_backend(backend_name) is not None


def get_backend_class(backend_name: str):
    """Get a backend class by name, loading it if necessary."""
    if backend_name in _backend_registry:
        return _backend_registry[backend_name]
    loaded = _load_backend(backend_name)
    if loaded is None and backend_name in _EXPERIMENTAL_BACKENDS:
        raise NotImplementedError(
            f"{backend_name} is not implemented and is not on the default backend path. "
            "Set MULTIMIND_EXPERIMENTAL_VECTOR_STORES=1 to load the stub, which still "
            "raises from every operation."
        )
    return loaded


# Create __all__ list
__all__ = [
    # Core classes
    "VectorStoreBackend",
    "VectorStoreConfig",
    "SearchResult",
    "VectorStoreType",
    "VectorStore",
    "VectorStoreFactory",
    # Utility functions
    "get_available_backends",
    "is_backend_available",
    "get_backend_class",
]

# Log summary
logger.info("📊 Vector store package loaded with lazy loading enabled")
logger.debug("Backends will be loaded only when requested")
