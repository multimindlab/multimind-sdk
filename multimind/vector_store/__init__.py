"""
Vector store package for managing vector storage and retrieval.
"""

import logging
import os
from .base import VectorStoreBackend, VectorStoreConfig, SearchResult, VectorStoreType
from .vector_store import VectorStore

# Configure logging to suppress warnings for optional backends
logger = logging.getLogger(__name__)

def _log_backend_warning(backend_name: str, error: Exception) -> None:
    """Log backend warning only if explicitly enabled."""
    # Check environment variable each time to allow runtime changes
    show_warnings = os.getenv('MULTIMIND_SHOW_BACKEND_WARNINGS', 'false').lower() == 'true'
    if show_warnings:
        logger.warning(f"{backend_name} backend not available - {str(error)}")
    else:
        # Use debug level to avoid console spam but keep it in logs
        logger.debug(f"{backend_name} backend not available - {str(error)}")

# Import all backend classes with graceful error handling
backend_classes = {}

def get_available_backends() -> list:
    """Get list of available vector store backends."""
    return list(backend_classes.keys())

def is_backend_available(backend_name: str) -> bool:
    """Check if a specific backend is available."""
    return backend_name in backend_classes

def get_backend_class(backend_name: str):
    """Get a backend class by name."""
    return backend_classes.get(backend_name)

try:
    from .faiss import FAISSBackend
    backend_classes['FAISSBackend'] = FAISSBackend
    logger.debug("✅ FAISS backend loaded successfully")
except (ImportError, Exception) as e:
    _log_backend_warning("FAISS", e)

try:
    from .chroma import ChromaBackend
    backend_classes['ChromaBackend'] = ChromaBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Chroma", e)

try:
    from .weaviate import WeaviateVectorStore
    backend_classes['WeaviateVectorStore'] = WeaviateVectorStore
except (ImportError, Exception) as e:
    _log_backend_warning("Weaviate", e)

try:
    from .qdrant import QdrantBackend
    backend_classes['QdrantBackend'] = QdrantBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Qdrant", e)

try:
    from .milvus import MilvusBackend
    backend_classes['MilvusBackend'] = MilvusBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Milvus", e)

try:
    from .pinecone import PineconeBackend
    backend_classes['PineconeBackend'] = PineconeBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Pinecone", e)

try:
    from .elasticsearch import ElasticsearchBackend
    backend_classes['ElasticsearchBackend'] = ElasticsearchBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Elasticsearch", e)

try:
    from .alibabacloud_opensearch import AlibabaCloudOpenSearchBackend
    backend_classes['AlibabaCloudOpenSearchBackend'] = AlibabaCloudOpenSearchBackend
except (ImportError, Exception) as e:
    _log_backend_warning("AlibabaCloud OpenSearch", e)

try:
    from .atlas import AtlasBackend
    backend_classes['AtlasBackend'] = AtlasBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Atlas", e)

try:
    from .awadb import AwaDBBackend
    backend_classes['AwaDBBackend'] = AwaDBBackend
except (ImportError, Exception) as e:
    _log_backend_warning("AwaDB", e)

try:
    from .azuresearch import AzureSearchBackend
    backend_classes['AzureSearchBackend'] = AzureSearchBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Azure Search", e)

try:
    from .bageldb import BagelDBBackend
    backend_classes['BagelDBBackend'] = BagelDBBackend
except (ImportError, Exception) as e:
    _log_backend_warning("BagelDB", e)

try:
    from .baiducloud_vector_search import BaiduCloudVectorSearchBackend
    backend_classes['BaiduCloudVectorSearchBackend'] = BaiduCloudVectorSearchBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Baidu Cloud Vector Search", e)

try:
    from .cassandra import CassandraBackend
    backend_classes['CassandraBackend'] = CassandraBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Cassandra", e)

try:
    from .clarifai import ClarifaiBackend
    backend_classes['ClarifaiBackend'] = ClarifaiBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Clarifai", e)

try:
    from .clickhouse import ClickHouseBackend
    backend_classes['ClickHouseBackend'] = ClickHouseBackend
except (ImportError, Exception) as e:
    _log_backend_warning("ClickHouse", e)

try:
    from .databricks_vector_search import DatabricksVectorSearchBackend
    backend_classes['DatabricksVectorSearchBackend'] = DatabricksVectorSearchBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Databricks Vector Search", e)

try:
    from .dashvector import DashVectorBackend
    backend_classes['DashVectorBackend'] = DashVectorBackend
except (ImportError, Exception) as e:
    _log_backend_warning("DashVector", e)

try:
    from .dingo import DingoDBBackend
    backend_classes['DingoDBBackend'] = DingoDBBackend
except (ImportError, Exception) as e:
    _log_backend_warning("DingoDB", e)

try:
    from .elastic_vector_search import ElasticVectorSearchBackend
    backend_classes['ElasticVectorSearchBackend'] = ElasticVectorSearchBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Elastic Vector Search", e)

try:
    from .hologres import HologresBackend
    backend_classes['HologresBackend'] = HologresBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Hologres", e)

try:
    from .lancedb import LanceDBBackend
    backend_classes['LanceDBBackend'] = LanceDBBackend
except (ImportError, Exception) as e:
    _log_backend_warning("LanceDB", e)

try:
    from .marqo import MarqoBackend
    backend_classes['MarqoBackend'] = MarqoBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Marqo", e)

try:
    from .meilisearch import MeiliSearchBackend
    backend_classes['MeiliSearchBackend'] = MeiliSearchBackend
except (ImportError, Exception) as e:
    _log_backend_warning("MeiliSearch", e)

try:
    from .mongodb_atlas import MongoDBAtlasBackend
    backend_classes['MongoDBAtlasBackend'] = MongoDBAtlasBackend
except (ImportError, Exception) as e:
    _log_backend_warning("MongoDB Atlas", e)

try:
    from .momento_vector_index import MomentoVectorIndexBackend
    backend_classes['MomentoVectorIndexBackend'] = MomentoVectorIndexBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Momento Vector Index", e)

try:
    from .neo4j_vector import Neo4jVectorBackend
    backend_classes['Neo4jVectorBackend'] = Neo4jVectorBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Neo4j Vector", e)

try:
    from .opensearch_vector_search import OpenSearchVectorBackend
    backend_classes['OpenSearchVectorBackend'] = OpenSearchVectorBackend
except (ImportError, Exception) as e:
    _log_backend_warning("OpenSearch Vector Search", e)

try:
    from .pgvector import PGVectorBackend
    backend_classes['PGVectorBackend'] = PGVectorBackend
except (ImportError, Exception) as e:
    _log_backend_warning("PGVector", e)

try:
    from .pgvecto_rs import PGVectoRSBackend
    backend_classes['PGVectoRSBackend'] = PGVectoRSBackend
except (ImportError, Exception) as e:
    _log_backend_warning("PGVectoRS", e)

try:
    from .pgembedding import PGEmbeddingBackend
    backend_classes['PGEmbeddingBackend'] = PGEmbeddingBackend
except (ImportError, Exception) as e:
    _log_backend_warning("PGEmbedding", e)

try:
    from .nucliadb import NucliaDBBackend
    backend_classes['NucliaDBBackend'] = NucliaDBBackend
except (ImportError, Exception) as e:
    _log_backend_warning("NucliaDB", e)

try:
    from .myscale import MyScaleBackend
    backend_classes['MyScaleBackend'] = MyScaleBackend
except (ImportError, Exception) as e:
    _log_backend_warning("MyScale", e)

try:
    from .matching_engine import MatchingEngineBackend
    backend_classes['MatchingEngineBackend'] = MatchingEngineBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Matching Engine", e)

try:
    from .llm_rails import LLMRailsBackend
    backend_classes['LLMRailsBackend'] = LLMRailsBackend
except (ImportError, Exception) as e:
    _log_backend_warning("LLM Rails", e)

try:
    from .hippo import HippoBackend
    backend_classes['HippoBackend'] = HippoBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Hippo", e)

try:
    from .epsilla import EpsillaBackend
    backend_classes['EpsillaBackend'] = EpsillaBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Epsilla", e)

try:
    from .deeplake import DeepLakeBackend
    backend_classes['DeepLakeBackend'] = DeepLakeBackend
except (ImportError, Exception) as e:
    _log_backend_warning("DeepLake", e)

try:
    from .azure_cosmos_db import AzureCosmosDBBackend
    backend_classes['AzureCosmosDBBackend'] = AzureCosmosDBBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Azure Cosmos DB", e)

try:
    from .annoy import AnnoyBackend
    backend_classes['AnnoyBackend'] = AnnoyBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Annoy", e)

try:
    from .astradb import AstraDBBackend
    backend_classes['AstraDBBackend'] = AstraDBBackend
except (ImportError, Exception) as e:
    _log_backend_warning("AstraDB", e)

try:
    from .analyticdb import AnalyticDBBackend
    backend_classes['AnalyticDBBackend'] = AnalyticDBBackend
except (ImportError, Exception) as e:
    _log_backend_warning("AnalyticDB", e)

try:
    from .sklearn import SklearnBackend
    backend_classes['SklearnBackend'] = SklearnBackend
except (ImportError, Exception) as e:
    _log_backend_warning("Sklearn", e)

try:
    from .singlestoredb import SingleStoreDBBackend
    backend_classes['SingleStoreDBBackend'] = SingleStoreDBBackend
except (ImportError, Exception) as e:
    _log_backend_warning("SingleStoreDB", e)

try:
    from .rocksetdb import RocksetDBBackend
    backend_classes['RocksetDBBackend'] = RocksetDBBackend
except (ImportError, Exception) as e:
    _log_backend_warning("RocksetDB", e)

try:
    from .sqlitevss import SQLiteVSSBackend
    backend_classes['SQLiteVSSBackend'] = SQLiteVSSBackend
except (ImportError, Exception) as e:
    _log_backend_warning("SQLiteVSS", e)

try:
    from .starrocks import StarRocksBackend
    backend_classes['StarRocksBackend'] = StarRocksBackend
except (ImportError, Exception) as e:
    _log_backend_warning("StarRocks", e)

try:
    from .supabase import SupabaseVectorStore
    backend_classes['SupabaseVectorStore'] = SupabaseVectorStore
except (ImportError, Exception) as e:
    _log_backend_warning("Supabase", e)

try:
    from .tair import TairVectorStore
    backend_classes['TairVectorStore'] = TairVectorStore
except (ImportError, Exception) as e:
    _log_backend_warning("Tair", e)

try:
    from .tigris import TigrisVectorStore
    backend_classes['TigrisVectorStore'] = TigrisVectorStore
except (ImportError, Exception) as e:
    _log_backend_warning("Tigris", e)

try:
    from .tiledb import TileDBVectorStore
    backend_classes['TileDBVectorStore'] = TileDBVectorStore
except (ImportError, Exception) as e:
    _log_backend_warning("TileDB", e)

try:
    from .timescalevector import TimescaleVectorStore
    backend_classes['TimescaleVectorStore'] = TimescaleVectorStore
except (ImportError, Exception) as e:
    _log_backend_warning("TimescaleVector", e)

try:
    from .tencentvectordb import TencentVectorDBVectorStore
    backend_classes['TencentVectorDBVectorStore'] = TencentVectorDBVectorStore
except (ImportError, Exception) as e:
    _log_backend_warning("TencentVectorDB", e)

try:
    from .usearch import USearchVectorStore
    backend_classes['USearchVectorStore'] = USearchVectorStore
except (ImportError, Exception) as e:
    _log_backend_warning("USearch", e)

try:
    from .vald import ValdVectorStore
    backend_classes['ValdVectorStore'] = ValdVectorStore
except (ImportError, Exception) as e:
    _log_backend_warning("Vald", e)

try:
    from .vectara import VectaraVectorStore
    backend_classes['VectaraVectorStore'] = VectaraVectorStore
except (ImportError, Exception) as e:
    _log_backend_warning("Vectara", e)

try:
    from .typesense import TypesenseVectorStore
    backend_classes['TypesenseVectorStore'] = TypesenseVectorStore
except (ImportError, Exception) as e:
    _log_backend_warning("Typesense", e)

try:
    from .xata import XataVectorStore
    backend_classes['XataVectorStore'] = XataVectorStore
except (ImportError, Exception) as e:
    _log_backend_warning("Xata", e)

try:
    from .zep import ZepVectorStore
    backend_classes['ZepVectorStore'] = ZepVectorStore
except (ImportError, Exception) as e:
    _log_backend_warning("Zep", e)

try:
    from .zilliz import ZillizVectorStore
    backend_classes['ZillizVectorStore'] = ZillizVectorStore
except (ImportError, Exception) as e:
    _log_backend_warning("Zilliz", e)

# Create __all__ list dynamically from available backends
__all__ = [
    # Core classes
    'VectorStoreBackend',
    'VectorStoreConfig',
    'SearchResult',
    'VectorStoreType',
    'VectorStore',
    # Utility functions
    'get_available_backends',
    'is_backend_available',
    'get_backend_class',
]

# Add available backends to __all__
__all__.extend(backend_classes.keys())

# Log summary of loaded backends
available_count = len(backend_classes)
logger.info(f"📊 Vector store backends loaded: {available_count} available")
if available_count > 0:
    logger.debug(f"Available backends: {', '.join(backend_classes.keys())}") 