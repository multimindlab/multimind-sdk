from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

class MilvusBackend(VectorStoreBackend):
    """
    Advanced Milvus Vector Store Backend.
    Features:
    - Dynamic schema: add custom fields to the collection
    - Partition management: create, drop, and use partitions
    - Custom search params: user-supplied at search time
    - Upsert support
    - Hybrid search: combine vector search with scalar (metadata) filtering
    - Advanced index types: user-supplied index params
    - Index management: create, drop, and list indexes
    """
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        db_name: Optional[str] = None,
        collection_name: str = "vectors",
        dim: int = 768,
        partition_name: Optional[str] = None,
        custom_fields: Optional[List[Dict[str, Any]]] = None,
        index_params: Optional[Dict[str, Any]] = None,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.host = host or os.environ.get("MILVUS_HOST", "localhost")
        self.port = port or os.environ.get("MILVUS_PORT", "19530")
        self.user = user or os.environ.get("MILVUS_USER")
        self.password = password or os.environ.get("MILVUS_PASSWORD")
        self.db_name = db_name or os.environ.get("MILVUS_DB_NAME", "default")
        self.collection_name = collection_name
        self.dim = dim
        self.partition_name = partition_name
        self.custom_fields = custom_fields or []
        self.index_params = index_params or {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            db_name=self.db_name
        )
        # Dynamic schema: add user-supplied fields
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        for f in self.custom_fields:
            # Example: {"name": "score", "dtype": DataType.FLOAT, "is_primary": False}
            fields.append(FieldSchema(**f))
        if not utility.has_collection(self.collection_name):
            schema = CollectionSchema(fields, description="Vector collection")
            Collection(self.collection_name, schema)
        self.collection = Collection(self.collection_name)
        # Advanced index types
        if not self.collection.has_index():
            self.collection.create_index(
                field_name="vector",
                index_params=self.index_params
            )
        self.collection.load()
        # Partition management
        if self.partition_name and self.partition_name not in self.collection.partitions:
            self.collection.create_partition(self.partition_name)

    async def add_vectors(self, vectors, metadatas, documents, ids=None, partition_name: Optional[str] = None, custom_fields_data: Optional[List[Dict[str, Any]]] = None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        # Dynamic schema: support custom fields
        data = [ids, vectors, metadatas]
        if self.custom_fields:
            for field in self.custom_fields:
                fname = field["name"]
                values = [d.get(fname) if d else None for d in (custom_fields_data or [{} for _ in range(n)])]
                data.append(values)
        loop = asyncio.get_event_loop()
        partition = partition_name or self.partition_name
        kwargs = {"partition_name": partition} if partition else {}
        await loop.run_in_executor(None, lambda: self.collection.insert(data, **kwargs))
        self.log_metrics('add_vectors', n)

    async def upsert_vectors(self, vectors, metadatas, documents, ids=None, partition_name: Optional[str] = None, custom_fields_data: Optional[List[Dict[str, Any]]] = None):
        ids = ids or [str(i) for i in range(len(vectors))]
        await self.delete_vectors(ids)
        await self.add_vectors(vectors, metadatas, documents, ids, partition_name, custom_fields_data)
        self.log_metrics('upsert_vectors', len(vectors))

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None, search_params: Optional[Dict[str, Any]] = None, partition_name: Optional[str] = None, hybrid_filter: Optional[str] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        search_params = search_params or self.index_params
        # Hybrid search: combine vector and scalar filtering
        expr = hybrid_filter
        if not expr and filter_criteria:
            expr = " and ".join([f"metadata['{k}'] == '{v}'" for k, v in filter_criteria.items()])
        partition = partition_name or self.partition_name
        kwargs = {"partition_names": [partition]} if partition else {}
        results = await loop.run_in_executor(
            None,
            lambda: self.collection.search(
                [query_vector],
                "vector",
                search_params,
                k,
                expr,
                output_fields=["id", "metadata"] + (metadata_fields or []),
                **kwargs
            )
        )
        search_results = []
        for hit in results[0]:
            search_results.append(
                SearchResult(
                    id=hit.id,
                    score=hit.distance,
                    metadata=hit.entity.get("metadata", {}),
                    document=""
                )
            )
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids, partition_name: Optional[str] = None):
        expr = f"id in [{', '.join([repr(i) for i in ids])}]"
        loop = asyncio.get_event_loop()
        partition = partition_name or self.partition_name
        kwargs = {"partition_name": partition} if partition else {}
        await loop.run_in_executor(None, lambda: self.collection.delete(expr, **kwargs))
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.collection.drop)
        # Recreate collection with dynamic schema and index
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        for f in self.custom_fields:
            fields.append(FieldSchema(**f))
        schema = CollectionSchema(fields, description="Vector collection")
        Collection(self.collection_name, schema)
        self.collection = Collection(self.collection_name)
        self.collection.create_index(
            field_name="vector",
            index_params=self.index_params
        )
        self.collection.load()
        self.log_metrics('clear', 1)

    async def create_partition(self, partition_name: str):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.collection.create_partition(partition_name))
        self.log_metrics('create_partition', partition_name)

    async def drop_partition(self, partition_name: str):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.collection.drop_partition(partition_name))
        self.log_metrics('drop_partition', partition_name)

    async def create_index(self, field_name: str, index_params: Dict[str, Any]):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.collection.create_index(field_name=field_name, index_params=index_params))
        self.log_metrics('create_index', field_name)

    async def drop_index(self, field_name: str):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.collection.drop_index(field_name=field_name))
        self.log_metrics('drop_index', field_name)

    async def persist(self, path):
        self.log_metrics('persist', 1)

    @classmethod
    async def load(cls, path, config):
        backend = cls(**config.connection_params)
        return backend

    def register_plugin(self, name: str, plugin: Callable):
        self.plugin_registry[name] = plugin

    async def _run_plugin(self, name: str, *args, **kwargs):
        if name in self.plugin_registry:
            if asyncio.iscoroutinefunction(self.plugin_registry[name]):
                await self.plugin_registry[name](*args, **kwargs)
            else:
                self.plugin_registry[name](*args, **kwargs)

    def log_metrics(self, metric_name: str, value: Any):
        if self.metrics_enabled:
            self.logger.info(f"[METRIC] {metric_name}: {value}")

    async def _with_retries(self, func, *args, **kwargs):
        retries = self.retry_policy.get('retries', 3)
        for attempt in range(retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error: {e}, attempt {attempt+1}/{retries}")
                if attempt == retries - 1:
                    raise 