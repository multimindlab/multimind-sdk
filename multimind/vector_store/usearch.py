import os
import logging
import asyncio
import numpy as np
import usearch
from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable

class USearchVectorStore(VectorStoreBackend):
    def __init__(
        self,
        index_path: Optional[str] = None,
        dim: int = 768,
        metric: str = "cosine",
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.index_path = index_path or os.environ.get("USEARCH_INDEX_PATH", "usearch_index")
        self.dim = dim
        self.metric = metric
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self.index = usearch.Index(
            metric=self.metric,
            ndim=self.dim,
            path=self.index_path
        )
        self._ensure_index()

    def _ensure_index(self):
        if not os.path.exists(self.index_path):
            self.index.save(self.index_path)

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or list(range(n))
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        loop = asyncio.get_event_loop()
        def _add():
            for i in range(n):
                self.index.add(ids[i], np.array(vectors[i], dtype=np.float32))
        await loop.run_in_executor(None, _add)
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        def _search():
            try:
                results = self.index.search(np.array(query_vector, dtype=np.float32), k)
            except Exception as e:
                self.logger.error(f"Search failed: {e}")
                results = []
            search_results = []
            for idx, score in results:
                search_results.append(SearchResult(
                    id=idx,
                    score=score,
                    metadata=None,
                    document=None
                ))
            return search_results
        search_results = await loop.run_in_executor(None, _search)
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        def _delete():
            for id_ in ids:
                self.index.remove(id_)
        await loop.run_in_executor(None, _delete)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        def _clear():
            self.index.clear()
        await loop.run_in_executor(None, _clear)
        self.log_metrics('clear', 1)

    async def persist(self, path):
        loop = asyncio.get_event_loop()
        def _persist():
            self.index.save(path)
        await loop.run_in_executor(None, _persist)
        self.log_metrics('persist', 1)

    @classmethod
    async def load(cls, path, config):
        backend = cls(**config.connection_params)
        backend.index.load(path)
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