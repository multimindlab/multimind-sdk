import os
import logging
import asyncio
import numpy as np
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable

class ZillizVectorStore(VectorStoreBackend):
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        collection: str = "vectors",
        dim: int = 768,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.host = host or os.environ.get("ZILLIZ_HOST", "localhost")
        self.port = port or int(os.environ.get("ZILLIZ_PORT", 19530))
        self.user = user or os.environ.get("ZILLIZ_USER", "")
        self.password = password or os.environ.get("ZILLIZ_PASSWORD", "")
        self.collection = collection
        self.dim = dim
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password
        )
        self._ensure_collection()

    def _ensure_collection(self):
        if self.collection not in [c.name for c in Collection.list()]:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=4096)
            ]
            schema = CollectionSchema(fields, description="Zilliz vector collection")
            Collection(self.collection, schema)

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        loop = asyncio.get_event_loop()
        def _add():
            col = Collection(self.collection)
            data = [ids, [list(map(float, v)) for v in vectors], metadatas, docs]
            col.insert(data)
        await loop.run_in_executor(None, _add)
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        def _search():
            col = Collection(self.collection)
            expr = None
            if filter_criteria:
                expr = " and ".join([f"metadata['{k}'] == '{v}'" for k, v in filter_criteria.items()])
            res = col.search(
                data=[list(map(float, query_vector))],
                anns_field="vector",
                param={"metric_type": "L2", "params": {"nprobe": 16}},
                limit=k,
                expr=expr,
                output_fields=["id", "metadata", "document"]
            )
            search_results = []
            for hits in res:
                for hit in hits:
                    search_results.append(SearchResult(
                        id=hit.entity.get("id"),
                        score=hit.distance,
                        metadata=hit.entity.get("metadata"),
                        document=hit.entity.get("document")
                    ))
            return search_results
        search_results = await loop.run_in_executor(None, _search)
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        def _delete():
            col = Collection(self.collection)
            expr = f"id in {[id_ for id_ in ids]}"
            col.delete(expr)
        await loop.run_in_executor(None, _delete)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        def _clear():
            col = Collection(self.collection)
            col.drop()
            self._ensure_collection()
        await loop.run_in_executor(None, _clear)
        self.log_metrics('clear', 1)

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