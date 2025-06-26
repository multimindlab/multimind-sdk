"""
Vector store package for managing vector storage and retrieval.
"""

from .base import VectorStoreBackend, VectorStoreConfig, SearchResult, VectorStoreType
from .vector_store import VectorStore

# Import all backend classes for direct usage
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

__all__ = [
    # Core classes
    'VectorStoreBackend',
    'VectorStoreConfig',
    'SearchResult',
    'VectorStoreType',
    'VectorStore',
    
    # All backend implementations
    'FAISSBackend',
    'ChromaBackend',
    'WeaviateVectorStore',
    'QdrantBackend',
    'MilvusBackend',
    'PineconeBackend',
    'ElasticsearchBackend',
    'AlibabaCloudOpenSearchBackend',
    'AtlasBackend',
    'AwaDBBackend',
    'AzureSearchBackend',
    'BagelDBBackend',
    'BaiduCloudVectorSearchBackend',
    'CassandraBackend',
    'ClarifaiBackend',
    'ClickHouseBackend',
    'DatabricksVectorSearchBackend',
    'DashVectorBackend',
    'DingoDBBackend',
    'ElasticVectorSearchBackend',
    'HologresBackend',
    'LanceDBBackend',
    'MarqoBackend',
    'MeiliSearchBackend',
    'MongoDBAtlasBackend',
    'MomentoVectorIndexBackend',
    'Neo4jVectorBackend',
    'OpenSearchVectorBackend',
    'PGVectorBackend',
    'PGVectoRSBackend',
    'PGEmbeddingBackend',
    'NucliaDBBackend',
    'MyScaleBackend',
    'MatchingEngineBackend',
    'LLMRailsBackend',
    'HippoBackend',
    'EpsillaBackend',
    'DeepLakeBackend',
    'AzureCosmosDBBackend',
    'AnnoyBackend',
    'AstraDBBackend',
    'AnalyticDBBackend',
    'SklearnBackend',
    'SingleStoreDBBackend',
    'RocksetDBBackend',
    'SQLiteVSSBackend',
    'StarRocksBackend',
    'SupabaseVectorStore',
    'TairVectorStore',
    'TigrisVectorStore',
    'TileDBVectorStore',
    'TimescaleVectorStore',
    'TencentVectorDBVectorStore',
    'USearchVectorStore',
    'ValdVectorStore',
    'VectaraVectorStore',
    'TypesenseVectorStore',
    'XataVectorStore',
    'ZepVectorStore',
    'ZillizVectorStore',
] 