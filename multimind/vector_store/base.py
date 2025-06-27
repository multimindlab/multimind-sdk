from typing import Any, Dict, List, Optional, Callable, Union, Set
import abc
from enum import Enum
import logging

class VectorStoreType(Enum):
    """
    Enumeration of supported vector store types with enhanced type safety and categorization.
    
    This enum provides type-safe access to all supported vector database backends,
    with helper methods for validation, categorization, and discovery.
    
    Categories:
    - IN_MEMORY: Fast, local, no persistence (FAISS, Sklearn, Annoy)
    - LOCAL_FILE: Local file-based storage (Chroma, LanceDB, DeepLake)
    - POSTGRESQL: PostgreSQL extensions (PGVector, PGVecto-RS, PGEmbedding)
    - CLOUD_SERVICES: Managed cloud services (Pinecone, Weaviate, Qdrant)
    - SEARCH_ENGINES: Full-text search engines (Elasticsearch, OpenSearch)
    - DATABASES: Traditional databases with vector support
    - CLOUD_PLATFORMS: Cloud platform services (Azure, AWS, GCP)
    - SPECIALIZED: Specialized vector solutions
    """
    
    # In-Memory Backends (Fast, local, no persistence)
    FAISS = "faiss"
    SKLEARN = "sklearn"
    ANNOY = "annoy"
    
    # Local File-Based Backends
    CHROMA = "chroma"
    LANCEDB = "lancedb"
    DEEPLAKE = "deeplake"
    SQLITEVSS = "sqlitevss"
    
    # PostgreSQL Extensions
    PGVECTOR = "pgvector"
    PGVECTO_RS = "pgvecto_rs"
    PGEMBEDDING = "pgembedding"
    
    # Cloud Services
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    PINECONE = "pinecone"
    VECTARA = "vectara"
    SUPABASE = "supabase"
    TYPESENSE = "typesense"
    MEILISEARCH = "meilisearch"
    MARQO = "marqo"
    USEARCH = "usearch"
    VALD = "vald"
    
    # Search Engines
    ELASTICSEARCH = "elasticsearch"
    ELASTIC_VECTOR_SEARCH = "elastic_vector_search"
    OPENSEARCH_VECTOR_SEARCH = "opensearch_vector_search"
    ALIBABACLOUD_OPENSEARCH = "alibabacloud_opensearch"
    
    # Databases
    CASSANDRA = "cassandra"
    CLICKHOUSE = "clickhouse"
    MONGODB_ATLAS = "mongodb_atlas"
    NEO4J_VECTOR = "neo4j_vector"
    SINGLESTOREDB = "singlestoredb"
    ROCKSETDB = "rocksetdb"
    STARROCKS = "starrocks"
    TIMESCALEVECTOR = "timescalevector"
    TILEDB = "tiledb"
    
    # Cloud Platforms
    AZURESEARCH = "azuresearch"
    AZURE_COSMOS_DB = "azure_cosmos_db"
    ATLAS = "atlas"
    ASTRADB = "astradb"
    HOLOGRES = "hologres"
    MYSCALE = "myscale"
    TENCENTVECTORDB = "tencentvectordb"
    BAIDUCLOUD_VECTOR_SEARCH = "baiducloud_vector_search"
    DATABRICKS_VECTOR_SEARCH = "databricks_vector_search"
    
    # Specialized
    MATCHING_ENGINE = "matching_engine"
    MOMENTO_VECTOR_INDEX = "momento_vector_index"
    LLM_RAILS = "llm_rails"
    CLARIFAI = "clarifai"
    DASHVECTOR = "dashvector"
    DINGO = "dingo"
    EPSILLA = "epsilla"
    HIPPO = "hippo"
    ANALYTICDB = "analyticdb"
    TAIR = "tair"
    TIGRIS = "tigris"
    XATA = "xata"
    ZEP = "zep"
    ZILLIZ = "zilliz"
    AWADB = "awadb"
    BAGELDB = "bageldb"
    NUCLIADB = "nucliadb"
    
    # Legacy/Deprecated (kept for backward compatibility)
    REDIS = "redis"
    POSTGRES = "postgres"
    
    @classmethod
    def get_in_memory_backends(cls) -> Set['VectorStoreType']:
        """Get all in-memory vector store types."""
        return {
            cls.FAISS, cls.SKLEARN, cls.ANNOY
        }
    
    @classmethod
    def get_local_file_backends(cls) -> Set['VectorStoreType']:
        """Get all local file-based vector store types."""
        return {
            cls.CHROMA, cls.LANCEDB, cls.DEEPLAKE, cls.SQLITEVSS
        }
    
    @classmethod
    def get_postgresql_backends(cls) -> Set['VectorStoreType']:
        """Get all PostgreSQL extension vector store types."""
        return {
            cls.PGVECTOR, cls.PGVECTO_RS, cls.PGEMBEDDING
        }
    
    @classmethod
    def get_cloud_service_backends(cls) -> Set['VectorStoreType']:
        """Get all cloud service vector store types."""
        return {
            cls.WEAVIATE, cls.QDRANT, cls.MILVUS, cls.PINECONE,
            cls.VECTARA, cls.SUPABASE, cls.TYPESENSE, cls.MEILISEARCH,
            cls.MARQO, cls.USEARCH, cls.VALD
        }
    
    @classmethod
    def get_search_engine_backends(cls) -> Set['VectorStoreType']:
        """Get all search engine vector store types."""
        return {
            cls.ELASTICSEARCH, cls.ELASTIC_VECTOR_SEARCH,
            cls.OPENSEARCH_VECTOR_SEARCH, cls.ALIBABACLOUD_OPENSEARCH
        }
    
    @classmethod
    def get_database_backends(cls) -> Set['VectorStoreType']:
        """Get all database vector store types."""
        return {
            cls.CASSANDRA, cls.CLICKHOUSE, cls.MONGODB_ATLAS,
            cls.NEO4J_VECTOR, cls.SINGLESTOREDB, cls.ROCKSETDB,
            cls.STARROCKS, cls.TIMESCALEVECTOR, cls.TILEDB
        }
    
    @classmethod
    def get_cloud_platform_backends(cls) -> Set['VectorStoreType']:
        """Get all cloud platform vector store types."""
        return {
            cls.AZURESEARCH, cls.AZURE_COSMOS_DB, cls.ATLAS,
            cls.ASTRADB, cls.HOLOGRES, cls.MYSCALE,
            cls.TENCENTVECTORDB, cls.BAIDUCLOUD_VECTOR_SEARCH,
            cls.DATABRICKS_VECTOR_SEARCH
        }
    
    @classmethod
    def get_specialized_backends(cls) -> Set['VectorStoreType']:
        """Get all specialized vector store types."""
        return {
            cls.MATCHING_ENGINE, cls.MOMENTO_VECTOR_INDEX,
            cls.LLM_RAILS, cls.CLARIFAI, cls.DASHVECTOR,
            cls.DINGO, cls.EPSILLA, cls.HIPPO, cls.ANALYTICDB,
            cls.TAIR, cls.TIGRIS, cls.XATA, cls.ZEP,
            cls.ZILLIZ, cls.AWADB, cls.BAGELDB, cls.NUCLIADB
        }
    
    @classmethod
    def get_all_backends(cls) -> Set['VectorStoreType']:
        """Get all vector store types."""
        return set(cls)
    
    @classmethod
    def get_backends_by_category(cls, category: str) -> Set['VectorStoreType']:
        """Get vector store types by category."""
        category_map = {
            "in_memory": cls.get_in_memory_backends(),
            "local_file": cls.get_local_file_backends(),
            "postgresql": cls.get_postgresql_backends(),
            "cloud_service": cls.get_cloud_service_backends(),
            "search_engine": cls.get_search_engine_backends(),
            "database": cls.get_database_backends(),
            "cloud_platform": cls.get_cloud_platform_backends(),
            "specialized": cls.get_specialized_backends(),
            "all": cls.get_all_backends()
        }
        return category_map.get(category.lower(), set())
    
    @classmethod
    def validate_store_type(cls, store_type: str) -> bool:
        """Validate if a store type string is supported."""
        try:
            cls(store_type)
            return True
        except ValueError:
            return False
    
    @classmethod
    def from_string(cls, store_type: str) -> 'VectorStoreType':
        """Create VectorStoreType from string with validation."""
        try:
            return cls(store_type)
        except ValueError:
            valid_types = [t.value for t in cls]
            raise ValueError(f"Invalid store type '{store_type}'. Valid types: {valid_types}")
    
    def is_in_memory(self) -> bool:
        """Check if this is an in-memory backend."""
        return self in self.get_in_memory_backends()
    
    def is_local_file(self) -> bool:
        """Check if this is a local file-based backend."""
        return self in self.get_local_file_backends()
    
    def is_postgresql(self) -> bool:
        """Check if this is a PostgreSQL extension backend."""
        return self in self.get_postgresql_backends()
    
    def is_cloud_service(self) -> bool:
        """Check if this is a cloud service backend."""
        return self in self.get_cloud_service_backends()
    
    def is_search_engine(self) -> bool:
        """Check if this is a search engine backend."""
        return self in self.get_search_engine_backends()
    
    def is_database(self) -> bool:
        """Check if this is a database backend."""
        return self in self.get_database_backends()
    
    def is_cloud_platform(self) -> bool:
        """Check if this is a cloud platform backend."""
        return self in self.get_cloud_platform_backends()
    
    def is_specialized(self) -> bool:
        """Check if this is a specialized backend."""
        return self in self.get_specialized_backends()
    
    def get_category(self) -> str:
        """Get the category of this vector store type."""
        if self.is_in_memory():
            return "in_memory"
        elif self.is_local_file():
            return "local_file"
        elif self.is_postgresql():
            return "postgresql"
        elif self.is_cloud_service():
            return "cloud_service"
        elif self.is_search_engine():
            return "search_engine"
        elif self.is_database():
            return "database"
        elif self.is_cloud_platform():
            return "cloud_platform"
        elif self.is_specialized():
            return "specialized"
        else:
            return "unknown"
    
    def get_description(self) -> str:
        """Get a human-readable description of this vector store type."""
        descriptions = {
            cls.FAISS: "Facebook AI Similarity Search - Fast in-memory vector search",
            cls.CHROMA: "ChromaDB - Local file-based vector database",
            cls.WEAVIATE: "Weaviate - Vector search engine with GraphQL API",
            cls.QDRANT: "Qdrant - Vector similarity search engine",
            cls.MILVUS: "Milvus - Open-source vector database",
            cls.PINECONE: "Pinecone - Managed vector database service",
            cls.ELASTICSEARCH: "Elasticsearch - Distributed search and analytics engine",
            cls.PGVECTOR: "PGVector - PostgreSQL extension for vector operations",
            cls.SKLEARN: "Scikit-learn - Machine learning library with vector search",
            cls.ANNOY: "Annoy - Approximate nearest neighbors library",
            # Add more descriptions as needed
        }
        return descriptions.get(self, f"{self.value} - Vector store backend")

class SearchResult:
    def __init__(self, id: str, vector: Any, metadata: Dict[str, Any], document: Any, score: float, explanation: Optional[Dict[str, Any]] = None):
        self.id = id
        self.vector = vector
        self.metadata = metadata
        self.document = document
        self.score = score
        self.explanation = explanation

class VectorStoreConfig:
    """
    Configuration class for vector store backends with enhanced type safety and validation.
    
    This class provides a type-safe way to configure vector store backends,
    with validation, helper methods, and support for different configuration patterns.
    """
    
    def __init__(self, connection_params: Dict[str, Any], store_type: Optional[VectorStoreType] = None):
        """
        Initialize vector store configuration.
        
        Args:
            connection_params: Dictionary of connection parameters
            store_type: Optional VectorStoreType enum value for type safety
        """
        self.connection_params = connection_params.copy() if connection_params else {}
        self._store_type = store_type
        
        # Validate configuration if store_type is provided
        if store_type:
            self._validate_config(store_type)
    
    @property
    def store_type(self) -> Optional[str]:
        """Get the store type as a string."""
        if self._store_type:
            return self._store_type.value
        return self.connection_params.get("store_type")
    
    @store_type.setter
    def store_type(self, value: Union[str, VectorStoreType]):
        """Set the store type with validation."""
        if isinstance(value, VectorStoreType):
            self._store_type = value
            self.connection_params["store_type"] = value.value
        elif isinstance(value, str):
            self._store_type = VectorStoreType.from_string(value)
            self.connection_params["store_type"] = value
        else:
            raise ValueError(f"store_type must be a string or VectorStoreType, got {type(value)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration parameter with a default value."""
        return self.connection_params.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration parameter."""
        self.connection_params[key] = value
    
    def has(self, key: str) -> bool:
        """Check if a configuration parameter exists."""
        return key in self.connection_params
    
    def get_required(self, key: str) -> Any:
        """Get a required configuration parameter, raising an error if not found."""
        if key not in self.connection_params:
            raise ValueError(f"Required configuration parameter '{key}' not found")
        return self.connection_params[key]
    
    def validate_required_params(self, required_params: List[str]) -> None:
        """Validate that all required parameters are present."""
        missing_params = [param for param in required_params if param not in self.connection_params]
        if missing_params:
            raise ValueError(f"Missing required configuration parameters: {missing_params}")
    
    def _validate_config(self, store_type: VectorStoreType) -> None:
        """Validate configuration for a specific store type."""
        # Define required parameters for each store type
        required_params = {
            VectorStoreType.FAISS: ["dimension"],
            VectorStoreType.CHROMA: ["persist_directory"],
            VectorStoreType.PINECONE: ["api_key", "environment", "index_name"],
            VectorStoreType.WEAVIATE: ["url"],
            VectorStoreType.QDRANT: ["url"],
            VectorStoreType.MILVUS: ["host", "port"],
            VectorStoreType.ELASTICSEARCH: ["hosts"],
            VectorStoreType.PGVECTOR: ["host", "port", "database", "user", "password"],
            VectorStoreType.SKLEARN: ["algorithm"],
            # Add more validation rules as needed
        }
        
        if store_type in required_params:
            self.validate_required_params(required_params[store_type])
    
    def get_store_type_enum(self) -> Optional[VectorStoreType]:
        """Get the store type as a VectorStoreType enum."""
        return self._store_type
    
    def is_valid(self) -> bool:
        """Check if the configuration is valid."""
        try:
            if self._store_type:
                self._validate_config(self._store_type)
            return True
        except ValueError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.connection_params.copy()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VectorStoreConfig':
        """Create VectorStoreConfig from dictionary."""
        store_type_str = config_dict.get("store_type")
        store_type = None
        if store_type_str:
            store_type = VectorStoreType.from_string(store_type_str)
        
        return cls(config_dict, store_type)
    
    @classmethod
    def create_faiss_config(cls, dimension: int, metric: str = "cosine", **kwargs) -> 'VectorStoreConfig':
        """Create a FAISS configuration."""
        config = {
            "store_type": VectorStoreType.FAISS.value,
            "dimension": dimension,
            "metric": metric,
            **kwargs
        }
        return cls(config, VectorStoreType.FAISS)
    
    @classmethod
    def create_chroma_config(cls, persist_directory: str, collection_name: str = "default", **kwargs) -> 'VectorStoreConfig':
        """Create a Chroma configuration."""
        config = {
            "store_type": VectorStoreType.CHROMA.value,
            "persist_directory": persist_directory,
            "collection_name": collection_name,
            **kwargs
        }
        return cls(config, VectorStoreType.CHROMA)
    
    @classmethod
    def create_pinecone_config(cls, api_key: str, environment: str, index_name: str, **kwargs) -> 'VectorStoreConfig':
        """Create a Pinecone configuration."""
        config = {
            "store_type": VectorStoreType.PINECONE.value,
            "api_key": api_key,
            "environment": environment,
            "index_name": index_name,
            **kwargs
        }
        return cls(config, VectorStoreType.PINECONE)
    
    @classmethod
    def create_weaviate_config(cls, url: str, **kwargs) -> 'VectorStoreConfig':
        """Create a Weaviate configuration."""
        config = {
            "store_type": VectorStoreType.WEAVIATE.value,
            "url": url,
            **kwargs
        }
        return cls(config, VectorStoreType.WEAVIATE)
    
    @classmethod
    def create_pgvector_config(cls, host: str, port: int, database: str, user: str, password: str, **kwargs) -> 'VectorStoreConfig':
        """Create a PGVector configuration."""
        config = {
            "store_type": VectorStoreType.PGVECTOR.value,
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
            **kwargs
        }
        return cls(config, VectorStoreType.PGVECTOR)
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        store_type_str = self.store_type or "unknown"
        return f"VectorStoreConfig(store_type='{store_type_str}', params={len(self.connection_params)} items)"
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return self.__repr__()

class VectorStoreBackend(abc.ABC):
    """
    Abstract base class for all vector store backends.
    
    This class defines the core interface that all vector store backends must implement.
    It focuses on the essential operations: initialization, adding vectors, searching,
    deleting, clearing, and persistence.
    """
    
    def __init__(self, config: VectorStoreConfig):
        """
        Initialize the vector store backend.
        
        Args:
            config: Configuration for the vector store backend
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abc.abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the vector store backend.
        
        This method should handle any setup required for the backend,
        such as creating connections, indexes, or loading data.
        """
        pass
    
    @abc.abstractmethod
    async def add_vectors(
        self, 
        vectors: List[List[float]], 
        metadatas: List[Dict[str, Any]], 
        documents: List[Dict[str, Any]], 
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add vectors to the vector store.
        
        Args:
            vectors: List of vector embeddings
            metadatas: List of metadata dictionaries for each vector
            documents: List of document dictionaries for each vector
            ids: Optional list of IDs for each vector. If not provided, 
                 the backend should generate appropriate IDs.
        """
        pass
    
    @abc.abstractmethod
    async def search(
        self, 
        query_vector: List[float], 
        k: int = 5, 
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: The query vector to search for
            k: Number of results to return
            filter_criteria: Optional metadata filters to apply
            
        Returns:
            List of SearchResult objects containing the most similar vectors
        """
        pass
    
    @abc.abstractmethod
    async def delete_vectors(self, ids: List[str]) -> None:
        """
        Delete vectors by their IDs.
        
        Args:
            ids: List of vector IDs to delete
        """
        pass
    
    @abc.abstractmethod
    async def clear(self) -> None:
        """
        Clear all vectors from the vector store.
        """
        pass
    
    @abc.abstractmethod
    async def persist(self, path: str) -> None:
        """
        Persist the vector store to disk.
        
        Args:
            path: Path where to save the vector store
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> 'VectorStoreBackend':
        """
        Load a vector store from disk.
        
        Args:
            path: Path where the vector store is saved
            config: Configuration for the vector store
            
        Returns:
            Loaded VectorStoreBackend instance
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the vector store backend.
        
        Returns:
            Dictionary containing backend information
        """
        return {
            "backend_type": self.__class__.__name__,
            "config": self.config.to_dict()
        }
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the vector store backend.
        
        Returns:
            True if the backend is healthy, False otherwise
        """
        try:
            # Basic health check - try to perform a simple operation
            await self.initialize()
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
