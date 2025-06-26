from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
import llmrails

class LLMRailsBackend(VectorStoreBackend):
    def __init__(
        self,
        api_key: Optional[str] = None,
        datastore_id: Optional[str] = None,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.api_key = api_key or os.environ.get("LLM_RAILS_API_KEY")
        self.datastore_id = datastore_id or os.environ.get("LLM_RAILS_DATASTORE_ID")
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        if not self.api_key or not self.datastore_id:
            raise ValueError("LLMRails API key and datastore_id must be provided.")
        self._client = llmrails.Client(self.api_key)

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        # LLMRails expects text, not raw vectors; use documents as input
        data = []
        for i, doc in enumerate(documents):
            entry = {"text": doc}
            if metadatas and i < len(metadatas):
                entry["metadata"] = metadatas[i]
            if ids and i < len(ids):
                entry["id"] = ids[i]
            data.append(entry)
        # LLMRails API is sync, so run in thread
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._client.add_texts, [d["text"] for d in data], self.datastore_id)
        self.log_metrics('add_vectors', len(data))

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        # LLMRails expects a query string, not a vector
        if not query_text:
            raise ValueError("query_text must be provided for LLMRails search.")
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._client.similarity_search, query_text, self.datastore_id, k)
        search_results = [
            SearchResult(
                id=str(i),
                score=getattr(r, 'score', 0.0),
                metadata=getattr(r, 'metadata', {}),
                document=getattr(r, 'page_content', str(r))
            ) for i, r in enumerate(results)
        ]
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        # LLMRails API does not support direct vector deletion; placeholder for future
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        # LLMRails API does not support clearing all vectors; placeholder for future
        self.log_metrics('clear', 1)

    async def persist(self, path):
        # LLMRails is managed; no-op
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