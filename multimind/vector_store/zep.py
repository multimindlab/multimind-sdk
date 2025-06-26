import os
import logging
import asyncio
import numpy as np
from zep_python import ZepClient
from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable

class ZepVectorStore(VectorStoreBackend):
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection: str = "vectors",
        dim: int = 768,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.api_url = api_url or os.environ.get("ZEP_API_URL")
        self.api_key = api_key or os.environ.get("ZEP_API_KEY")
        self.collection = collection
        self.dim = dim
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self.client = ZepClient(base_url=self.api_url, api_key=self.api_key)
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            colls = self.client.collections.list()
            if self.collection not in [c['name'] for c in colls]:
                self.client.collections.create({
                    'name': self.collection,
                    'dimension': self.dim
                })
        except Exception as e:
            self.logger.warning(f"Collection ensure failed or already exists: {e}")

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        loop = asyncio.get_event_loop()
        def _add():
            for i in range(n):
                self.client.documents.add(
                    collection=self.collection,
                    document_id=ids[i],
                    vector=list(map(float, vectors[i])),
                    metadata=metadatas[i],
                    text=docs[i]
                )
        await loop.run_in_executor(None, _add)
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        def _search():
            try:
                res = self.client.documents.search(
                    collection=self.collection,
                    vector=list(map(float, query_vector)),
                    k=k,
                    filter=filter_criteria
                )
                results = res.get('results', [])
            except Exception as e:
                self.logger.error(f"Search failed: {e}")
                results = []
            search_results = []
            for row in results:
                search_results.append(SearchResult(
                    id=row.get('id'),
                    score=row.get('score', 0),
                    metadata=row.get('metadata'),
                    document=row.get('text')
                ))
            return search_results
        search_results = await loop.run_in_executor(None, _search)
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        def _delete():
            for id_ in ids:
                self.client.documents.delete(collection=self.collection, document_id=id_)
        await loop.run_in_executor(None, _delete)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        def _clear():
            self.client.collections.delete(self.collection)
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