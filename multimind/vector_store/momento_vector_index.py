from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
import momento

class MomentoVectorIndexBackend(VectorStoreBackend):
    def __init__(
        self,
        auth_token: Optional[str] = None,
        index_name: str = "vectors",
        endpoint: Optional[str] = None,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.auth_token = auth_token or os.environ.get("MOMENTO_AUTH_TOKEN")
        self.index_name = index_name
        self.endpoint = endpoint or os.environ.get("MOMENTO_ENDPOINT")
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self._client = momento.Client(self.auth_token, endpoint=self.endpoint) if self.endpoint else momento.Client(self.auth_token)
        # Ensure index exists (Momento auto-creates on upsert)

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        # Momento upserts vectors with metadata
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        loop = asyncio.get_event_loop()
        for i in range(n):
            doc = documents[i] if documents else ""
            meta = metadatas[i] if metadatas else {}
            await loop.run_in_executor(
                None,
                lambda: self._client.vector.upsert(
                    self.index_name,
                    ids[i],
                    vectors[i],
                    metadata=meta,
                    document=doc
                )
            )
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        # Momento supports vector search with optional metadata filtering
        filter_expr = None
        if filter_criteria:
            filter_expr = " AND ".join([f"metadata.{k} == '{v}'" for k, v in filter_criteria.items()])
        result = await loop.run_in_executor(
            None,
            lambda: self._client.vector.search(
                self.index_name,
                query_vector,
                top_k=k,
                filter=filter_expr
            )
        )
        hits = result.get('matches', [])
        search_results = [
            SearchResult(
                id=hit.get('id'),
                score=hit.get('score', 0.0),
                metadata=hit.get('metadata', {}),
                document=hit.get('document', "")
            ) for hit in hits
        ]
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        for id_ in ids:
            await loop.run_in_executor(None, lambda: self._client.vector.delete(self.index_name, id_))
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        # Momento does not have a direct clear; delete all by listing
        loop = asyncio.get_event_loop()
        all_ids = await loop.run_in_executor(None, lambda: [item['id'] for item in self._client.vector.list(self.index_name)])
        for id_ in all_ids:
            await loop.run_in_executor(None, lambda: self._client.vector.delete(self.index_name, id_))
        self.log_metrics('clear', len(all_ids))

    async def persist(self, path):
        # Momento is managed and persistent
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