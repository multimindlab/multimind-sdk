from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests.auth import HTTPBasicAuth
import numpy as np

class OpenSearchVectorBackend(VectorStoreBackend):
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        index: str = "vectors",
        vector_field: str = "vector",
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.host = host or os.environ.get("OPENSEARCH_HOST", "localhost")
        self.port = port or int(os.environ.get("OPENSEARCH_PORT", 9200))
        self.user = user or os.environ.get("OPENSEARCH_USER")
        self.password = password or os.environ.get("OPENSEARCH_PASSWORD")
        self.index = index
        self.vector_field = vector_field
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self._client = OpenSearch(
            hosts=[{"host": self.host, "port": self.port}],
            http_auth=HTTPBasicAuth(self.user, self.password) if self.user and self.password else None,
            use_ssl=False,
            verify_certs=False,
            connection_class=RequestsHttpConnection
        )
        # Ensure index exists
        if not self._client.indices.exists(index=self.index):
            self._client.indices.create(index=self.index, body={
                "mappings": {
                    "properties": {
                        self.vector_field: {"type": "knn_vector", "dimension": 768},
                        "metadata": {"type": "object"},
                        "document": {"type": "text"}
                    }
                }
            })

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        loop = asyncio.get_event_loop()
        def _add():
            for i in range(n):
                self._client.index(
                    index=self.index,
                    id=ids[i],
                    body={
                        self.vector_field: list(map(float, vectors[i])),
                        "metadata": metadatas[i],
                        "document": docs[i]
                    }
                )
        await loop.run_in_executor(None, _add)
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        query = {
            "size": k,
            "query": {
                "knn": {
                    self.vector_field: {
                        "vector": list(map(float, query_vector)),
                        "k": k
                    }
                }
            }
        }
        if filter_criteria:
            query["query"] = {
                "bool": {
                    "must": [
                        {"knn": {self.vector_field: {"vector": list(map(float, query_vector)), "k": k}}},
                        {"match": filter_criteria}
                    ]
                }
            }
        result = await loop.run_in_executor(None, lambda: self._client.search(index=self.index, body=query))
        hits = result.get('hits', {}).get('hits', [])
        search_results = [
            SearchResult(
                id=hit.get('_id'),
                score=hit.get('_score', 0.0),
                metadata=hit.get('_source', {}).get('metadata', {}),
                document=hit.get('_source', {}).get('document', "")
            ) for hit in hits
        ]
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        def _delete():
            for id_ in ids:
                self._client.delete(index=self.index, id=id_, ignore=[404])
        await loop.run_in_executor(None, _delete)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._client.delete_by_query(index=self.index, body={"query": {"match_all": {}}}))
        self.log_metrics('clear', 1)

    async def persist(self, path):
        # OpenSearch is persistent by default
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