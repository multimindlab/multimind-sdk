import os
import logging
import asyncio
import numpy as np
import weaviate
from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable

class WeaviateVectorStore(VectorStoreBackend):
    def __init__(
        self,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
        class_name: str = "Vector",
        dim: int = 768,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.host = host or os.environ.get("WEAVIATE_HOST", "http://localhost:8080")
        self.api_key = api_key or os.environ.get("WEAVIATE_API_KEY")
        self.class_name = class_name
        self.dim = dim
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self.client = weaviate.Client(self.host, auth_client_secret=self.api_key)
        self._ensure_class()

    def _ensure_class(self):
        if not self.client.schema.exists(self.class_name):
            self.client.schema.create_class({
                "class": self.class_name,
                "vectorIndexType": "hnsw",
                "vectorizer": "none",
                "properties": [
                    {"name": "vector", "dataType": ["number[]"]},
                    {"name": "metadata", "dataType": ["object"]},
                    {"name": "document", "dataType": ["text"]}
                ]
            })

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        loop = asyncio.get_event_loop()
        def _add():
            for i in range(n):
                data = {
                    "vector": list(map(float, vectors[i])),
                    "metadata": metadatas[i],
                    "document": docs[i]
                }
                self.client.data_object.create(data, self.class_name, uuid=ids[i])
        await loop.run_in_executor(None, _add)
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        def _search():
            try:
                res = self.client.query.get(self.class_name, ["vector", "metadata", "document"]).with_near_vector({"vector": list(map(float, query_vector))}).with_limit(k).do()
                results = res.get("data", {}).get("Get", {}).get(self.class_name, [])
            except Exception as e:
                self.logger.error(f"Search failed: {e}")
                results = []
            search_results = []
            for row in results:
                search_results.append(SearchResult(
                    id=row.get("_additional", {}).get("id"),
                    score=row.get("_additional", {}).get("certainty", 0),
                    metadata=row.get("metadata"),
                    document=row.get("document")
                ))
            return search_results
        search_results = await loop.run_in_executor(None, _search)
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        def _delete():
            for id_ in ids:
                self.client.data_object.delete(uuid=id_, class_name=self.class_name)
        await loop.run_in_executor(None, _delete)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        def _clear():
            self.client.schema.delete_class(self.class_name)
            self._ensure_class()
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