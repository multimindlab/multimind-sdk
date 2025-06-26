from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
import marqo

class MarqoBackend(VectorStoreBackend):
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        index_name: str = "default-index",
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.api_key = api_key or os.environ.get("MARQO_API_KEY")
        self.endpoint = endpoint or os.environ.get("MARQO_ENDPOINT", "http://localhost:8882")
        self.index_name = index_name
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self._client = marqo.Client(url=self.endpoint, api_key=self.api_key)
        # Ensure index exists
        if self.index_name not in [idx['index_name'] for idx in self._client.get_indexes()['results']]:
            self._client.create_index(self.index_name)
        self._index = self._client.index(self.index_name)

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        # Marqo expects documents with text and optional metadata
        docs = []
        for i, doc in enumerate(documents):
            entry = {"_id": ids[i] if ids else str(i), "text": doc}
            if metadatas and i < len(metadatas):
                entry.update(metadatas[i])
            docs.append(entry)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._index.add_documents, docs)
        self.log_metrics('add_vectors', len(docs))

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        # Marqo supports both vector and text search; prefer query_text if provided
        loop = asyncio.get_event_loop()
        if query_text:
            result = await loop.run_in_executor(None, self._index.search, query_text, {"limit": k})
        else:
            # If only vector is provided, use vector search (requires Marqo v1.3+)
            result = await loop.run_in_executor(None, self._index.search, query_vector, {"limit": k, "search_method": "VECTOR"})
        hits = result.get('hits', [])
        search_results = [
            SearchResult(
                id=hit.get('_id'),
                score=hit.get('_score', 0.0),
                metadata={k: v for k, v in hit.items() if k not in ['_id', '_score', 'text']},
                document=hit.get('text', "")
            ) for hit in hits
        ]
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._index.delete_documents, ids)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        # Delete all documents in the index
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._index.delete)
        self.log_metrics('clear', 1)

    async def persist(self, path):
        # Marqo is persistent by default; this is a no-op
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