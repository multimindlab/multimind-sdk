import os
import logging
import asyncio
import numpy as np
from xata.client import XataClient
from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable

class XataVectorStore(VectorStoreBackend):
    def __init__(
        self,
        db_url: Optional[str] = None,
        api_key: Optional[str] = None,
        table: str = "vectors",
        dim: int = 768,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.db_url = db_url or os.environ.get("XATA_DB_URL")
        self.api_key = api_key or os.environ.get("XATA_API_KEY")
        self.table = table
        self.dim = dim
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self.client = XataClient(db_url=self.db_url, api_key=self.api_key)
        self._ensure_table()

    def _ensure_table(self):
        try:
            tables = self.client.tables().get()["tables"]
            if self.table not in [t["name"] for t in tables]:
                self.client.table(self.table).create({
                    "columns": [
                        {"name": "id", "type": "string"},
                        {"name": "vector", "type": "float[]", "size": self.dim},
                        {"name": "metadata", "type": "object"},
                        {"name": "document", "type": "text"}
                    ]
                })
        except Exception as e:
            self.logger.warning(f"Table ensure failed or already exists: {e}")

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        loop = asyncio.get_event_loop()
        def _add():
            for i in range(n):
                self.client.records().insert(self.table, {
                    "id": ids[i],
                    "vector": list(map(float, vectors[i])),
                    "metadata": metadatas[i],
                    "document": docs[i]
                })
        await loop.run_in_executor(None, _add)
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        def _search():
            try:
                res = self.client.records().search(self.table, {
                    "vector": list(map(float, query_vector)),
                    "k": k,
                    "filter": filter_criteria or {}
                })
                results = res.get("records", [])
            except Exception as e:
                self.logger.error(f"Search failed: {e}")
                results = []
            search_results = []
            for row in results:
                search_results.append(SearchResult(
                    id=row.get("id"),
                    score=row.get("score", 0),
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
                self.client.records().delete(self.table, id_)
        await loop.run_in_executor(None, _delete)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        def _clear():
            self.client.table(self.table).delete()
            self._ensure_table()
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