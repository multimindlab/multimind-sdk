import os
import logging
import asyncio
from tair import Tair
import numpy as np
from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable

class TairVectorStore(VectorStoreBackend):
    """Tair Vector Store Backend."""
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        password: Optional[str] = None,
        index_name: str = "tair_vectors",
        dim: int = 768,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.host = host or os.environ.get("TAIR_HOST", "localhost")
        self.port = port or int(os.environ.get("TAIR_PORT", 6379))
        self.password = password or os.environ.get("TAIR_PASSWORD", None)
        self.index_name = index_name
        self.dim = dim
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self.client = Tair(host=self.host, port=self.port, password=self.password)
        self._ensure_index()

    def _ensure_index(self):
        # Create index if not exists
        try:
            self.client.tvs_create_index(self.index_name, dim=self.dim, distance_type="l2")
        except Exception as e:
            self.logger.warning(f"Index ensure failed or already exists: {e}")

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        loop = asyncio.get_event_loop()
        def _add():
            for i in range(n):
                self.client.tvs_hset(
                    self.index_name,
                    ids[i],
                    vector=np.array(vectors[i], dtype=np.float32).tobytes(),
                    payload={"metadata": metadatas[i], "document": docs[i]}
                )
        await loop.run_in_executor(None, _add)
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        def _search():
            try:
                res = self.client.tvs_knnsearch(
                    self.index_name,
                    np.array(query_vector, dtype=np.float32).tobytes(),
                    k,
                    with_payload=True
                )
                results = res["result"] if isinstance(res, dict) and "result" in res else []
            except Exception as e:
                self.logger.error(f"Search failed: {e}")
                results = []
            search_results = []
            for row in results:
                meta = row.get("payload", {})
                search_results.append(SearchResult(
                    id=row.get("key"),
                    score=row.get("score", 0),
                    metadata=meta.get("metadata"),
                    document=meta.get("document")
                ))
            return search_results
        search_results = await loop.run_in_executor(None, _search)
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        def _delete():
            for id_ in ids:
                self.client.tvs_hdel(self.index_name, id_)
        await loop.run_in_executor(None, _delete)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        def _clear():
            self.client.delete(self.index_name)
            self._ensure_index()
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