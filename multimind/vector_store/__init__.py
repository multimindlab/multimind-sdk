"""
Vector store package for managing vector storage and retrieval.
"""

import logging
from .base import VectorStoreBackend, VectorStoreConfig, SearchResult, VectorStoreType
from .vector_store import VectorStore

# Import all backend classes with graceful error handling
backend_classes = {}

try:
    from .faiss import FAISSBackend
    backend_classes['FAISSBackend'] = FAISSBackend
except (ImportError, Exception) as e:
    logging.warning(f"FAISS backend not available - {str(e)}")

try:
    from .chroma import ChromaBackend
    backend_classes['ChromaBackend'] = ChromaBackend
except (ImportError, Exception) as e:
    logging.warning(f"Chroma backend not available - {str(e)}")

try:
    from .weaviate import WeaviateVectorStore
    backend_classes['WeaviateVectorStore'] = WeaviateVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Weaviate backend not available - {str(e)}")

try:
    from .qdrant import QdrantBackend
    backend_classes['QdrantBackend'] = QdrantBackend
except (ImportError, Exception) as e:
    logging.warning(f"Qdrant backend not available - {str(e)}")

try:
    from .milvus import MilvusBackend
    backend_classes['MilvusBackend'] = MilvusBackend
except (ImportError, Exception) as e:
    logging.warning(f"Milvus backend not available - {str(e)}")

try:
    from .pinecone import PineconeBackend
    backend_classes['PineconeBackend'] = PineconeBackend
except (ImportError, Exception) as e:
    logging.warning(f"Pinecone backend not available - {str(e)}")

try:
    from .elasticsearch import ElasticsearchBackend
    backend_classes['ElasticsearchBackend'] = ElasticsearchBackend
except (ImportError, Exception) as e:
    logging.warning(f"Elasticsearch backend not available - {str(e)}")

try:
    from .alibabacloud_opensearch import AlibabaCloudOpenSearchBackend
    backend_classes['AlibabaCloudOpenSearchBackend'] = AlibabaCloudOpenSearchBackend
except (ImportError, Exception) as e:
    logging.warning(f"AlibabaCloud OpenSearch backend not available - {str(e)}")

try:
    from .atlas import AtlasBackend
    backend_classes['AtlasBackend'] = AtlasBackend
except (ImportError, Exception) as e:
    logging.warning(f"Atlas backend not available - {str(e)}")

try:
    from .awadb import AwaDBBackend
    backend_classes['AwaDBBackend'] = AwaDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"AwaDB backend not available - {str(e)}")

try:
    from .azuresearch import AzureSearchBackend
    backend_classes['AzureSearchBackend'] = AzureSearchBackend
except (ImportError, Exception) as e:
    logging.warning(f"Azure Search backend not available - {str(e)}")

try:
    from .bageldb import BagelDBBackend
    backend_classes['BagelDBBackend'] = BagelDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"BagelDB backend not available - {str(e)}")

try:
    from .baiducloud_vector_search import BaiduCloudVectorSearchBackend
    backend_classes['BaiduCloudVectorSearchBackend'] = BaiduCloudVectorSearchBackend
except (ImportError, Exception) as e:
    logging.warning(f"Baidu Cloud Vector Search backend not available - {str(e)}")

try:
    from .cassandra import CassandraBackend
    backend_classes['CassandraBackend'] = CassandraBackend
except (ImportError, Exception) as e:
    logging.warning(f"Cassandra backend not available - {str(e)}")

try:
    from .clarifai import ClarifaiBackend
    backend_classes['ClarifaiBackend'] = ClarifaiBackend
except (ImportError, Exception) as e:
    logging.warning(f"Clarifai backend not available - {str(e)}")

try:
    from .clickhouse import ClickHouseBackend
    backend_classes['ClickHouseBackend'] = ClickHouseBackend
except (ImportError, Exception) as e:
    logging.warning(f"ClickHouse backend not available - {str(e)}")

try:
    from .databricks_vector_search import DatabricksVectorSearchBackend
    backend_classes['DatabricksVectorSearchBackend'] = DatabricksVectorSearchBackend
except (ImportError, Exception) as e:
    logging.warning(f"Databricks Vector Search backend not available - {str(e)}")

try:
    from .dashvector import DashVectorBackend
    backend_classes['DashVectorBackend'] = DashVectorBackend
except (ImportError, Exception) as e:
    logging.warning(f"DashVector backend not available - {str(e)}")

try:
    from .dingo import DingoDBBackend
    backend_classes['DingoDBBackend'] = DingoDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"DingoDB backend not available - {str(e)}")

try:
    from .elastic_vector_search import ElasticVectorSearchBackend
    backend_classes['ElasticVectorSearchBackend'] = ElasticVectorSearchBackend
except (ImportError, Exception) as e:
    logging.warning(f"Elastic Vector Search backend not available - {str(e)}")

try:
    from .hologres import HologresBackend
    backend_classes['HologresBackend'] = HologresBackend
except (ImportError, Exception) as e:
    logging.warning(f"Hologres backend not available - {str(e)}")

try:
    from .lancedb import LanceDBBackend
    backend_classes['LanceDBBackend'] = LanceDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"LanceDB backend not available - {str(e)}")

try:
    from .marqo import MarqoBackend
    backend_classes['MarqoBackend'] = MarqoBackend
except (ImportError, Exception) as e:
    logging.warning(f"Marqo backend not available - {str(e)}")

try:
    from .meilisearch import MeiliSearchBackend
    backend_classes['MeiliSearchBackend'] = MeiliSearchBackend
except (ImportError, Exception) as e:
    logging.warning(f"MeiliSearch backend not available - {str(e)}")

try:
    from .mongodb_atlas import MongoDBAtlasBackend
    backend_classes['MongoDBAtlasBackend'] = MongoDBAtlasBackend
except (ImportError, Exception) as e:
    logging.warning(f"MongoDB Atlas backend not available - {str(e)}")

try:
    from .momento_vector_index import MomentoVectorIndexBackend
    backend_classes['MomentoVectorIndexBackend'] = MomentoVectorIndexBackend
except (ImportError, Exception) as e:
    logging.warning(f"Momento Vector Index backend not available - {str(e)}")

try:
    from .neo4j_vector import Neo4jVectorBackend
    backend_classes['Neo4jVectorBackend'] = Neo4jVectorBackend
except (ImportError, Exception) as e:
    logging.warning(f"Neo4j Vector backend not available - {str(e)}")

try:
    from .opensearch_vector_search import OpenSearchVectorBackend
    backend_classes['OpenSearchVectorBackend'] = OpenSearchVectorBackend
except (ImportError, Exception) as e:
    logging.warning(f"OpenSearch Vector Search backend not available - {str(e)}")

try:
    from .pgvector import PGVectorBackend
    backend_classes['PGVectorBackend'] = PGVectorBackend
except (ImportError, Exception) as e:
    logging.warning(f"PGVector backend not available - {str(e)}")

try:
    from .pgvecto_rs import PGVectoRSBackend
    backend_classes['PGVectoRSBackend'] = PGVectoRSBackend
except (ImportError, Exception) as e:
    logging.warning(f"PGVectoRS backend not available - {str(e)}")

try:
    from .pgembedding import PGEmbeddingBackend
    backend_classes['PGEmbeddingBackend'] = PGEmbeddingBackend
except (ImportError, Exception) as e:
    logging.warning(f"PGEmbedding backend not available - {str(e)}")

try:
    from .nucliadb import NucliaDBBackend
    backend_classes['NucliaDBBackend'] = NucliaDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"NucliaDB backend not available - {str(e)}")

try:
    from .myscale import MyScaleBackend
    backend_classes['MyScaleBackend'] = MyScaleBackend
except (ImportError, Exception) as e:
    logging.warning(f"MyScale backend not available - {str(e)}")

try:
    from .matching_engine import MatchingEngineBackend
    backend_classes['MatchingEngineBackend'] = MatchingEngineBackend
except (ImportError, Exception) as e:
    logging.warning(f"Matching Engine backend not available - {str(e)}")

try:
    from .llm_rails import LLMRailsBackend
    backend_classes['LLMRailsBackend'] = LLMRailsBackend
except (ImportError, Exception) as e:
    logging.warning(f"LLM Rails backend not available - {str(e)}")

try:
    from .hippo import HippoBackend
    backend_classes['HippoBackend'] = HippoBackend
except (ImportError, Exception) as e:
    logging.warning(f"Hippo backend not available - {str(e)}")

try:
    from .epsilla import EpsillaBackend
    backend_classes['EpsillaBackend'] = EpsillaBackend
except (ImportError, Exception) as e:
    logging.warning(f"Epsilla backend not available - {str(e)}")

try:
    from .deeplake import DeepLakeBackend
    backend_classes['DeepLakeBackend'] = DeepLakeBackend
except (ImportError, Exception) as e:
    logging.warning(f"DeepLake backend not available - {str(e)}")

try:
    from .azure_cosmos_db import AzureCosmosDBBackend
    backend_classes['AzureCosmosDBBackend'] = AzureCosmosDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"Azure Cosmos DB backend not available - {str(e)}")

try:
    from .annoy import AnnoyBackend
    backend_classes['AnnoyBackend'] = AnnoyBackend
except (ImportError, Exception) as e:
    logging.warning(f"Annoy backend not available - {str(e)}")

try:
    from .astradb import AstraDBBackend
    backend_classes['AstraDBBackend'] = AstraDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"AstraDB backend not available - {str(e)}")

try:
    from .analyticdb import AnalyticDBBackend
    backend_classes['AnalyticDBBackend'] = AnalyticDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"AnalyticDB backend not available - {str(e)}")

try:
    from .sklearn import SklearnBackend
    backend_classes['SklearnBackend'] = SklearnBackend
except (ImportError, Exception) as e:
    logging.warning(f"Sklearn backend not available - {str(e)}")

try:
    from .singlestoredb import SingleStoreDBBackend
    backend_classes['SingleStoreDBBackend'] = SingleStoreDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"SingleStoreDB backend not available - {str(e)}")

try:
    from .rocksetdb import RocksetDBBackend
    backend_classes['RocksetDBBackend'] = RocksetDBBackend
except (ImportError, Exception) as e:
    logging.warning(f"RocksetDB backend not available - {str(e)}")

try:
    from .sqlitevss import SQLiteVSSBackend
    backend_classes['SQLiteVSSBackend'] = SQLiteVSSBackend
except (ImportError, Exception) as e:
    logging.warning(f"SQLiteVSS backend not available - {str(e)}")

try:
    from .starrocks import StarRocksBackend
    backend_classes['StarRocksBackend'] = StarRocksBackend
except (ImportError, Exception) as e:
    logging.warning(f"StarRocks backend not available - {str(e)}")

try:
    from .supabase import SupabaseVectorStore
    backend_classes['SupabaseVectorStore'] = SupabaseVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Supabase backend not available - {str(e)}")

try:
    from .tair import TairVectorStore
    backend_classes['TairVectorStore'] = TairVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Tair backend not available - {str(e)}")

try:
    from .tigris import TigrisVectorStore
    backend_classes['TigrisVectorStore'] = TigrisVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Tigris backend not available - {str(e)}")

try:
    from .tiledb import TileDBVectorStore
    backend_classes['TileDBVectorStore'] = TileDBVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"TileDB backend not available - {str(e)}")

try:
    from .timescalevector import TimescaleVectorStore
    backend_classes['TimescaleVectorStore'] = TimescaleVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"TimescaleVector backend not available - {str(e)}")

try:
    from .tencentvectordb import TencentVectorDBVectorStore
    backend_classes['TencentVectorDBVectorStore'] = TencentVectorDBVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"TencentVectorDB backend not available - {str(e)}")

try:
    from .usearch import USearchVectorStore
    backend_classes['USearchVectorStore'] = USearchVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"USearch backend not available - {str(e)}")

try:
    from .vald import ValdVectorStore
    backend_classes['ValdVectorStore'] = ValdVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Vald backend not available - {str(e)}")

try:
    from .vectara import VectaraVectorStore
    backend_classes['VectaraVectorStore'] = VectaraVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Vectara backend not available - {str(e)}")

try:
    from .typesense import TypesenseVectorStore
    backend_classes['TypesenseVectorStore'] = TypesenseVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Typesense backend not available - {str(e)}")

try:
    from .xata import XataVectorStore
    backend_classes['XataVectorStore'] = XataVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Xata backend not available - {str(e)}")

try:
    from .zep import ZepVectorStore
    backend_classes['ZepVectorStore'] = ZepVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Zep backend not available - {str(e)}")

try:
    from .zilliz import ZillizVectorStore
    backend_classes['ZillizVectorStore'] = ZillizVectorStore
except (ImportError, Exception) as e:
    logging.warning(f"Zilliz backend not available - {str(e)}")

# Create __all__ list dynamically from available backends
__all__ = [
    # Core classes
    'VectorStoreBackend',
    'VectorStoreConfig',
    'SearchResult',
    'VectorStoreType',
    'VectorStore',
]

# Add available backends to __all__
__all__.extend(backend_classes.keys()) 