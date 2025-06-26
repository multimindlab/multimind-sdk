import os
import logging
import asyncio
import numpy as np
from tcvectordb.client import VectorDBClient
from tcvectordb.model import InsertRequest, QueryRequest, DeleteRequest
from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable

class TencentVectorDBVectorStore(VectorStoreBackend):
    """Tencent VectorDB Vector Store Backend."""
    def __init__(
        self,
        endpoint: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "test_db",
        collection: str = "vectors",
        dim: int = 768,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.endpoint = endpoint or os.environ.get("TENCENT_VECTORDB_ENDPOINT")
        self.username = username or os.environ.get("TENCENT_VECTORDB_USERNAME")
        self.password = password or os.environ.get("TENCENT_VECTORDB_PASSWORD")
        self.database = database
        self.collection = collection
        self.dim = dim
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self.client = VectorDBClient(self.endpoint, self.username, self.password)
        self._ensure_collection()

    def _ensure_collection(self):
        # Create database and collection if not exists
        try:
            dbs = self.client.list_databases()
            if self.database not in [db.name for db in dbs]:
                self.client.create_database(self.database)
            colls = self.client.list_collections(self.database)
            if self.collection not in [c.name for c in colls]:
                self.client.create_collection(self.database, self.collection, dim=self.dim)
        except Exception as e:
            self.logger.warning(f"Collection ensure failed or already exists: {e}")

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        loop = asyncio.get_event_loop()
        def _add():
            reqs = []
            for i in range(n):
                reqs.append(InsertRequest(
                    id=ids[i],
                    vector=list(map(float, vectors[i])),
                    metadata=metadatas[i],
                    document=docs[i]
                ))
            self.client.insert(self.database, self.collection, reqs)
        await loop.run_in_executor(None, _add)
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        def _search():
            query = QueryRequest(
                vector=list(map(float, query_vector)),
                topk=k,
                filter=filter_criteria or {},
                return_metadata=True,
                return_document=True
            )
            try:
                res = self.client.query(self.database, self.collection, query)
                results = res.results if hasattr(res, 'results') else []
            except Exception as e:
                self.logger.error(f"Search failed: {e}")
                results = []
            search_results = []
            for row in results:
                search_results.append(SearchResult(
                    id=row.id,
                    score=row.score,
                    metadata=row.metadata,
                    document=row.document
                ))
            return search_results
        search_results = await loop.run_in_executor(None, _search)
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        def _delete():
            reqs = [DeleteRequest(id=id_) for id_ in ids]
            self.client.delete(self.database, self.collection, reqs)
        await loop.run_in_executor(None, _delete)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        def _clear():
            self.client.delete_collection(self.database, self.collection)
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