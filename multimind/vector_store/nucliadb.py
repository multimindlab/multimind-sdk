from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
from nucliadb_sdk import NucliaDB

class NucliaDBBackend(VectorStoreBackend):
    def __init__(
        self,
        host: Optional[str] = None,
        key: Optional[str] = None,
        kb: Optional[str] = None,
        resource_type: str = "vector",
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.host = host or os.environ.get("NUCLIADB_HOST", "https://nucliadb.cloud")
        self.key = key or os.environ.get("NUCLIADB_KEY")
        self.kb = kb or os.environ.get("NUCLIADB_KB")
        self.resource_type = resource_type
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self._client = NucliaDB(self.host, self.key)
        self._kb = self._client.kb(self.kb)

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        loop = asyncio.get_event_loop()
        for i in range(n):
            await loop.run_in_executor(
                None,
                lambda: self._kb.add_resource(
                    id=ids[i],
                    type=self.resource_type,
                    vectors=[vectors[i]],
                    metadata=metadatas[i],
                    text=docs[i]
                )
            )
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        # NucliaDB supports vector search with optional metadata filtering
        filter_expr = filter_criteria or {}
        result = await loop.run_in_executor(
            None,
            lambda: self._kb.search_vectors(
                vectors=[query_vector],
                top_k=k,
                filter=filter_expr
            )
        )
        hits = result.get('results', [])
        search_results = [
            SearchResult(
                id=hit.get('id'),
                score=hit.get('score', 0.0),
                metadata=hit.get('metadata', {}),
                document=hit.get('text', "")
            ) for hit in hits
        ]
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        for id_ in ids:
            await loop.run_in_executor(None, lambda: self._kb.delete_resource(id_))
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        # NucliaDB does not have a direct clear; delete all by listing
        loop = asyncio.get_event_loop()
        all_resources = await loop.run_in_executor(None, lambda: self._kb.list_resources())
        all_ids = [res['id'] for res in all_resources.get('resources', [])]
        for id_ in all_ids:
            await loop.run_in_executor(None, lambda: self._kb.delete_resource(id_))
        self.log_metrics('clear', len(all_ids))

    async def persist(self, path):
        # NucliaDB is managed and persistent
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