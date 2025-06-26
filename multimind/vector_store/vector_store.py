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

# Import all backend implementations
from .faiss import FAISSBackend
from .chroma import ChromaBackend
from .weaviate import WeaviateVectorStore
from .qdrant import QdrantBackend
from .milvus import MilvusBackend
from .pinecone import PineconeBackend
from .elasticsearch import ElasticsearchBackend
from .alibabacloud_opensearch import AlibabaCloudOpenSearchBackend
from .atlas import AtlasBackend
from .awadb import AwaDBBackend
from .azuresearch import AzureSearchBackend
from .bageldb import BagelDBBackend
from .baiducloud_vector_search import BaiduCloudVectorSearchBackend
from .cassandra import CassandraBackend
from .clarifai import ClarifaiBackend
from .clickhouse import ClickHouseBackend
from .databricks_vector_search import DatabricksVectorSearchBackend
from .dashvector import DashVectorBackend
from .dingo import DingoDBBackend
from .elastic_vector_search import ElasticVectorSearchBackend
from .hologres import HologresBackend
from .lancedb import LanceDBBackend
from .marqo import MarqoBackend
from .meilisearch import MeiliSearchBackend
from .mongodb_atlas import MongoDBAtlasBackend
from .momento_vector_index import MomentoVectorIndexBackend
from .neo4j_vector import Neo4jVectorBackend
from .opensearch_vector_search import OpenSearchVectorBackend
from .pgvector import PGVectorBackend
from .pgvecto_rs import PGVectoRSBackend
from .pgembedding import PGEmbeddingBackend
from .nucliadb import NucliaDBBackend
from .myscale import MyScaleBackend
from .matching_engine import MatchingEngineBackend
from .llm_rails import LLMRailsBackend
from .hippo import HippoBackend
from .epsilla import EpsillaBackend
from .deeplake import DeepLakeBackend
from .azure_cosmos_db import AzureCosmosDBBackend
from .annoy import AnnoyBackend
from .astradb import AstraDBBackend
from .analyticdb import AnalyticDBBackend
from .sklearn import SklearnBackend
from .singlestoredb import SingleStoreDBBackend
from .rocksetdb import RocksetDBBackend
from .sqlitevss import SQLiteVSSBackend
from .starrocks import StarRocksBackend
from .supabase import SupabaseVectorStore
from .tair import TairVectorStore
from .tigris import TigrisVectorStore
from .tiledb import TileDBVectorStore
from .timescalevector import TimescaleVectorStore
from .tencentvectordb import TencentVectorDBVectorStore
from .usearch import USearchVectorStore
from .vald import ValdVectorStore
from .vectara import VectaraVectorStore
from .typesense import TypesenseVectorStore
from .xata import XataVectorStore
from .zep import ZepVectorStore
from .zilliz import ZillizVectorStore

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
            VectorStoreType.WEAVIATE: WeaviateVectorStore,
            VectorStoreType.QDRANT: QdrantBackend,
            VectorStoreType.MILVUS: MilvusBackend,
            VectorStoreType.PINECONE: PineconeBackend,
            VectorStoreType.ELASTICSEARCH: ElasticsearchBackend,
            VectorStoreType.ALIBABACLOUD_OPENSEARCH: AlibabaCloudOpenSearchBackend,
            VectorStoreType.ATLAS: AtlasBackend,
            VectorStoreType.AWADB: AwaDBBackend,
            VectorStoreType.AZURESEARCH: AzureSearchBackend,
            VectorStoreType.BAGELDB: BagelDBBackend,
            VectorStoreType.BAIDUCLOUD_VECTOR_SEARCH: BaiduCloudVectorSearchBackend,
            VectorStoreType.CASSANDRA: CassandraBackend,
            VectorStoreType.CLARIFAI: ClarifaiBackend,
            VectorStoreType.CLICKHOUSE: ClickHouseBackend,
            VectorStoreType.DATABRICKS_VECTOR_SEARCH: DatabricksVectorSearchBackend,
            VectorStoreType.DASHVECTOR: DashVectorBackend,
            VectorStoreType.DINGO: DingoDBBackend,
            VectorStoreType.ELASTIC_VECTOR_SEARCH: ElasticVectorSearchBackend,
            VectorStoreType.HOLOGRES: HologresBackend,
            VectorStoreType.LANCEDB: LanceDBBackend,
            VectorStoreType.MARQO: MarqoBackend,
            VectorStoreType.MEILISEARCH: MeiliSearchBackend,
            VectorStoreType.MONGODB_ATLAS: MongoDBAtlasBackend,
            VectorStoreType.MOMENTO_VECTOR_INDEX: MomentoVectorIndexBackend,
            VectorStoreType.NEO4J_VECTOR: Neo4jVectorBackend,
            VectorStoreType.OPENSEARCH_VECTOR_SEARCH: OpenSearchVectorBackend,
            VectorStoreType.PGVECTOR: PGVectorBackend,
            VectorStoreType.PGVECTO_RS: PGVectoRSBackend,
            VectorStoreType.PGEMBEDDING: PGEmbeddingBackend,
            VectorStoreType.NUCLIADB: NucliaDBBackend,
            VectorStoreType.MYSCALE: MyScaleBackend,
            VectorStoreType.MATCHING_ENGINE: MatchingEngineBackend,
            VectorStoreType.LLM_RAILS: LLMRailsBackend,
            VectorStoreType.HIPPO: HippoBackend,
            VectorStoreType.EPSILLA: EpsillaBackend,
            VectorStoreType.DEEPLAKE: DeepLakeBackend,
            VectorStoreType.AZURE_COSMOS_DB: AzureCosmosDBBackend,
            VectorStoreType.ANNOY: AnnoyBackend,
            VectorStoreType.ASTRADB: AstraDBBackend,
            VectorStoreType.ANALYTICDB: AnalyticDBBackend,
            VectorStoreType.SKLEARN: SklearnBackend,
            VectorStoreType.SINGLESTOREDB: SingleStoreDBBackend,
            VectorStoreType.ROCKSETDB: RocksetDBBackend,
            VectorStoreType.SQLITEVSS: SQLiteVSSBackend,
            VectorStoreType.STARROCKS: StarRocksBackend,
            VectorStoreType.SUPABASE: SupabaseVectorStore,
            VectorStoreType.TAIR: TairVectorStore,
            VectorStoreType.TIGRIS: TigrisVectorStore,
            VectorStoreType.TILEDB: TileDBVectorStore,
            VectorStoreType.TIMESCALEVECTOR: TimescaleVectorStore,
            VectorStoreType.TENCENTVECTORDB: TencentVectorDBVectorStore,
            VectorStoreType.USEARCH: USearchVectorStore,
            VectorStoreType.VALD: ValdVectorStore,
            VectorStoreType.VECTARA: VectaraVectorStore,
            VectorStoreType.TYPESENSE: TypesenseVectorStore,
            VectorStoreType.XATA: XataVectorStore,
            VectorStoreType.ZEP: ZepVectorStore,
            VectorStoreType.ZILLIZ: ZillizVectorStore,
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