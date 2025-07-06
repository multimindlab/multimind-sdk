"""
Main vector store implementation that manages different backends.
this file that provides the unified abstraction for all vector store backends.
It allows users to switch between databases seamlessly by specifying the backend 
in the config, without changing their code.
All backend implementations (e.g., Milvus, Pinecone, Qdrant, etc.) are mapped in 
this file, so the user can select any supported backend via configuration.
"""

import logging
from typing import List, Dict, Any, Optional

from .base import VectorStoreBackend, VectorStoreConfig, SearchResult, VectorStoreType

# Import all backend implementations with graceful error handling
backend_classes = {}

try:
    from .faiss import FAISSBackend
    backend_classes[VectorStoreType.FAISS] = FAISSBackend
except (ImportError, Exception) as e:
    logging.warning(f"FAISS backend not available - {str(e)}")

try:
    from .chroma import ChromaBackend
    backend_classes[VectorStoreType.CHROMA] = ChromaBackend
except (ImportError, Exception) as e:
    logging.warning(f"Chroma backend not available - {str(e)}")

try:
    from .weaviate import WeaviateVectorStore
    backend_classes[VectorStoreType.WEAVIATE] = WeaviateVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Weaviate backend not available - {str(e)}")

try:
    from .qdrant import QdrantBackend
    backend_classes[VectorStoreType.QDRANT] = QdrantBackend
except (ImportError, Exception) as e:
    logging.warning(f"Qdrant backend not available - {str(e)}")

try:
    from .milvus import MilvusBackend
    backend_classes[VectorStoreType.MILVUS] = MilvusBackend
except (ImportError, Exception) as e:
    logging.warning(f"Milvus backend not available - {str(e)}")

try:
    from .pinecone import PineconeBackend
    backend_classes[VectorStoreType.PINECONE] = PineconeBackend
except (ImportError, Exception) as e:
    logging.warning(f"Pinecone backend not available - {str(e)}")

try:
    from .elasticsearch import ElasticsearchBackend
    backend_classes[VectorStoreType.ELASTICSEARCH] = ElasticsearchBackend
except (ImportError, Exception) as e:
    logging.warning(f"Elasticsearch backend not available - {str(e)}")

try:
    from .alibabacloud_opensearch import AlibabaCloudOpenSearchBackend
    backend_classes[VectorStoreType.ALIBABACLOUD_OPENSEARCH] = AlibabaCloudOpenSearchBackend
except (ImportError, Exception) as e:
    logging.warning(f"AlibabaCloud OpenSearch backend not available - {str(e)}")

try:
    from .atlas import AtlasBackend
    backend_classes[VectorStoreType.ATLAS] = AtlasBackend
except (ImportError, Exception) as e:
    logging.warning(f"Atlas backend not available - {str(e)}")

try:
    from .awadb import AwaDBBackend
    backend_classes[VectorStoreType.AWADB] = AwaDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"AwaDB backend not available - {str(e)}")

try:
    from .azuresearch import AzureSearchBackend
    backend_classes[VectorStoreType.AZURESEARCH] = AzureSearchBackend
except (ImportError, Exception) as e:
    logging.warning(f"Azure Search backend not available - {str(e)}")

try:
    from .bageldb import BagelDBBackend
    backend_classes[VectorStoreType.BAGELDB] = BagelDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"BagelDB backend not available - {str(e)}")

try:
    from .baiducloud_vector_search import BaiduCloudVectorSearchBackend
    backend_classes[VectorStoreType.BAIDUCLOUD_VECTOR_SEARCH] = BaiduCloudVectorSearchBackend
except (ImportError, Exception) as e:
    logging.warning(f"Baidu Cloud Vector Search backend not available - {str(e)}")

try:
    from .cassandra import CassandraBackend
    backend_classes[VectorStoreType.CASSANDRA] = CassandraBackend
except (ImportError, Exception) as e:
    logging.warning(f"Cassandra backend not available - {str(e)}")

try:
    from .clarifai import ClarifaiBackend
    backend_classes[VectorStoreType.CLARIFAI] = ClarifaiBackend
except (ImportError, Exception) as e:
    logging.warning(f"Clarifai backend not available - {str(e)}")

try:
    from .clickhouse import ClickHouseBackend
    backend_classes[VectorStoreType.CLICKHOUSE] = ClickHouseBackend
except (ImportError, Exception) as e:
    logging.warning(f"ClickHouse backend not available - {str(e)}")

try:
    from .databricks_vector_search import DatabricksVectorSearchBackend
    backend_classes[VectorStoreType.DATABRICKS_VECTOR_SEARCH] = DatabricksVectorSearchBackend
except (ImportError, Exception) as e:
    logging.warning(f"Databricks Vector Search backend not available - {str(e)}")

try:
    from .dashvector import DashVectorBackend
    backend_classes[VectorStoreType.DASHVECTOR] = DashVectorBackend
except (ImportError, Exception) as e:
    logging.warning(f"DashVector backend not available - {str(e)}")

try:
    from .dingo import DingoDBBackend
    backend_classes[VectorStoreType.DINGO] = DingoDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"DingoDB backend not available - {str(e)}")

try:
    from .elastic_vector_search import ElasticVectorSearchBackend
    backend_classes[VectorStoreType.ELASTIC_VECTOR_SEARCH] = ElasticVectorSearchBackend
except (ImportError, Exception) as e:
    logging.warning(f"Elastic Vector Search backend not available - {str(e)}")

try:
    from .hologres import HologresBackend
    backend_classes[VectorStoreType.HOLOGRES] = HologresBackend
except (ImportError, Exception) as e:
    logging.warning(f"Hologres backend not available - {str(e)}")

try:
    from .lancedb import LanceDBBackend
    backend_classes[VectorStoreType.LANCEDB] = LanceDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"LanceDB backend not available - {str(e)}")

try:
    from .marqo import MarqoBackend
    backend_classes[VectorStoreType.MARQO] = MarqoBackend
except (ImportError, Exception) as e:
    logging.warning(f"Marqo backend not available - {str(e)}")

try:
    from .meilisearch import MeiliSearchBackend
    backend_classes[VectorStoreType.MEILISEARCH] = MeiliSearchBackend
except (ImportError, Exception) as e:
    logging.warning(f"MeiliSearch backend not available - {str(e)}")

try:
    from .mongodb_atlas import MongoDBAtlasBackend
    backend_classes[VectorStoreType.MONGODB_ATLAS] = MongoDBAtlasBackend
except (ImportError, Exception) as e:
    logging.warning(f"MongoDB Atlas backend not available - {str(e)}")

try:
    from .momento_vector_index import MomentoVectorIndexBackend
    backend_classes[VectorStoreType.MOMENTO_VECTOR_INDEX] = MomentoVectorIndexBackend
except (ImportError, Exception) as e:
    logging.warning(f"Momento Vector Index backend not available - {str(e)}")

try:
    from .neo4j_vector import Neo4jVectorBackend
    backend_classes[VectorStoreType.NEO4J_VECTOR] = Neo4jVectorBackend
except (ImportError, Exception) as e:
    logging.warning(f"Neo4j Vector backend not available - {str(e)}")

try:
    from .opensearch_vector_search import OpenSearchVectorBackend
    backend_classes[VectorStoreType.OPENSEARCH_VECTOR_SEARCH] = OpenSearchVectorBackend
except (ImportError, Exception) as e:
    logging.warning(f"OpenSearch Vector Search backend not available - {str(e)}")

try:
    from .pgvector import PGVectorBackend
    backend_classes[VectorStoreType.PGVECTOR] = PGVectorBackend
except (ImportError, Exception) as e:
    logging.warning(f"PGVector backend not available - {str(e)}")

try:
    from .pgvecto_rs import PGVectoRSBackend
    backend_classes[VectorStoreType.PGVECTO_RS] = PGVectoRSBackend
except (ImportError, Exception) as e:
    logging.warning(f"PGVectoRS backend not available - {str(e)}")

try:
    from .pgembedding import PGEmbeddingBackend
    backend_classes[VectorStoreType.PGEMBEDDING] = PGEmbeddingBackend
except (ImportError, Exception) as e:
    logging.warning(f"PGEmbedding backend not available - {str(e)}")

try:
    from .nucliadb import NucliaDBBackend
    backend_classes[VectorStoreType.NUCLIADB] = NucliaDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"NucliaDB backend not available - {str(e)}")

try:
    from .myscale import MyScaleBackend
    backend_classes[VectorStoreType.MYSCALE] = MyScaleBackend
except (ImportError, Exception) as e:
    logging.warning(f"MyScale backend not available - {str(e)}")

try:
    from .matching_engine import MatchingEngineBackend
    backend_classes[VectorStoreType.MATCHING_ENGINE] = MatchingEngineBackend
except (ImportError, Exception) as e:
    logging.warning(f"Matching Engine backend not available - {str(e)}")

try:
    from .llm_rails import LLMRailsBackend
    backend_classes[VectorStoreType.LLM_RAILS] = LLMRailsBackend
except (ImportError, Exception) as e:
    logging.warning(f"LLM Rails backend not available - {str(e)}")

try:
    from .hippo import HippoBackend
    backend_classes[VectorStoreType.HIPPO] = HippoBackend
except (ImportError, Exception) as e:
    logging.warning(f"Hippo backend not available - {str(e)}")

try:
    from .epsilla import EpsillaBackend
    backend_classes[VectorStoreType.EPSILLA] = EpsillaBackend
except (ImportError, Exception) as e:
    logging.warning(f"Epsilla backend not available - {str(e)}")

try:
    from .deeplake import DeepLakeBackend
    backend_classes[VectorStoreType.DEEPLAKE] = DeepLakeBackend
except (ImportError, Exception) as e:
    logging.warning(f"DeepLake backend not available - {str(e)}")

try:
    from .azure_cosmos_db import AzureCosmosDBBackend
    backend_classes[VectorStoreType.AZURE_COSMOS_DB] = AzureCosmosDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"Azure Cosmos DB backend not available - {str(e)}")

try:
    from .annoy import AnnoyBackend
    backend_classes[VectorStoreType.ANNOY] = AnnoyBackend
except (ImportError, Exception) as e:
    logging.warning(f"Annoy backend not available - {str(e)}")

try:
    from .astradb import AstraDBBackend
    backend_classes[VectorStoreType.ASTRADB] = AstraDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"AstraDB backend not available - {str(e)}")

try:
    from .analyticdb import AnalyticDBBackend
    backend_classes[VectorStoreType.ANALYTICDB] = AnalyticDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"AnalyticDB backend not available - {str(e)}")

try:
    from .sklearn import SklearnBackend
    backend_classes[VectorStoreType.SKLEARN] = SklearnBackend
except (ImportError, Exception) as e:
    logging.warning(f"Sklearn backend not available - {str(e)}")

try:
    from .singlestoredb import SingleStoreDBBackend
    backend_classes[VectorStoreType.SINGLESTOREDB] = SingleStoreDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"SingleStoreDB backend not available - {str(e)}")

try:
    from .rocksetdb import RocksetDBBackend
    backend_classes[VectorStoreType.ROCKSETDB] = RocksetDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"RocksetDB backend not available - {str(e)}")

try:
    from .sqlitevss import SQLiteVSSBackend
    backend_classes[VectorStoreType.SQLITEVSS] = SQLiteVSSBackend
except (ImportError, Exception) as e:
    logging.warning(f"SQLiteVSS backend not available - {str(e)}")

try:
    from .starrocks import StarRocksBackend
    backend_classes[VectorStoreType.STARROCKS] = StarRocksBackend
except (ImportError, Exception) as e:
    logging.warning(f"StarRocks backend not available - {str(e)}")

try:
    from .supabase import SupabaseVectorStore
    backend_classes[VectorStoreType.SUPABASE] = SupabaseVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Supabase backend not available - {str(e)}")

try:
    from .tair import TairVectorStore
    backend_classes[VectorStoreType.TAIR] = TairVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Tair backend not available - {str(e)}")

try:
    from .tigris import TigrisVectorStore
    backend_classes[VectorStoreType.TIGRIS] = TigrisVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Tigris backend not available - {str(e)}")

try:
    from .tiledb import TileDBVectorStore
    backend_classes[VectorStoreType.TILEDB] = TileDBVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"TileDB backend not available - {str(e)}")

try:
    from .timescalevector import TimescaleVectorStore
    backend_classes[VectorStoreType.TIMESCALEVECTOR] = TimescaleVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"TimescaleVector backend not available - {str(e)}")

try:
    from .tencentvectordb import TencentVectorDBVectorStore
    backend_classes[VectorStoreType.TENCENTVECTORDB] = TencentVectorDBVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"TencentVectorDB backend not available - {str(e)}")

try:
    from .usearch import USearchVectorStore
    backend_classes[VectorStoreType.USEARCH] = USearchVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"USearch backend not available - {str(e)}")

try:
    from .vald import ValdVectorStore
    backend_classes[VectorStoreType.VALD] = ValdVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Vald backend not available - {str(e)}")

try:
    from .vectara import VectaraVectorStore
    backend_classes[VectorStoreType.VECTARA] = VectaraVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Vectara backend not available - {str(e)}")

try:
    from .typesense import TypesenseVectorStore
    backend_classes[VectorStoreType.TYPESENSE] = TypesenseVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Typesense backend not available - {str(e)}")

try:
    from .xata import XataVectorStore
    backend_classes[VectorStoreType.XATA] = XataVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Xata backend not available - {str(e)}")

try:
    from .zep import ZepVectorStore
    backend_classes[VectorStoreType.ZEP] = ZepVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Zep backend not available - {str(e)}")

try:
    from .zilliz import ZillizVectorStore
    backend_classes[VectorStoreType.ZILLIZ] = ZillizVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Zilliz backend not available - {str(e)}")

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
        
        backend_class = backend_classes.get(store_type)
        if not backend_class:
            available_backends = [str(backend) for backend in backend_classes.keys()]
            raise ValueError(
                f"Unsupported vector store type: {store_type}. "
                f"Available backends: {available_backends}"
            )
        
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