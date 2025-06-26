from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
import rockset
import numpy as np

class RocksetDBBackend(VectorStoreBackend):
    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        workspace: str = "commons",
        collection: str = "vectors",
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.api_key = api_key or os.environ.get("ROCKSET_API_KEY")
        self.host = host or os.environ.get("ROCKSET_HOST", "https://api.rs2.usw2.rockset.com")
        self.workspace = workspace
        self.collection = collection
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self._client = rockset.Client(api_key=self.api_key, host=self.host)
        # Ensure collection exists
        try:
            self._client.Collection.retrieve(self.workspace, self.collection)
        except rockset.exceptions.NotFoundException:
            self._client.Collection.create(self.workspace, name=self.collection)

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        items = []
        for i in range(n):
            item = {
                "id": ids[i],
                "vector": list(map(float, vectors[i])),
                "metadata": metadatas[i],
                "document": docs[i]
            }
            items.append(item)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._client.Documents.add(self.workspace, self.collection, data=items))
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        # Rockset does not natively support vector search, but you can use SQL UDFs for similarity
        loop = asyncio.get_event_loop()
        filter_sql = ""
        if filter_criteria:
            filter_clauses = [f"metadata['{k}'] = '{v}'" for k, v in filter_criteria.items()]
            filter_sql = " AND ".join(filter_clauses)
        where_clause = f"WHERE {filter_sql}" if filter_sql else ""
        query_vec = np.array(query_vector, dtype=np.float32)
        sql = f"""
            SELECT id, vector, metadata, document,
                1 - (vector <=> {list(query_vec)}) AS score
            FROM {self.workspace}.{self.collection}
            {where_clause}
            ORDER BY score DESC
            LIMIT {k}
        """
        def _search():
            return self._client.Query.query(sql=sql).results
        results = await loop.run_in_executor(None, _search)
        search_results = [
            SearchResult(
                id=row.get('id'),
                score=row.get('score', 0.0),
                metadata=row.get('metadata', {}),
                document=row.get('document', "")
            ) for row in results
        ]
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        # Rockset does not support direct delete by ID; use SQL
        loop = asyncio.get_event_loop()
        ids_list = ",".join([f"'" + str(i) + "'" for i in ids])
        sql = f"DELETE FROM {self.workspace}.{self.collection} WHERE id IN ({ids_list})"
        await loop.run_in_executor(None, lambda: self._client.Query.query(sql=sql))
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        # Truncate collection using SQL
        loop = asyncio.get_event_loop()
        sql = f"DELETE FROM {self.workspace}.{self.collection}"
        await loop.run_in_executor(None, lambda: self._client.Query.query(sql=sql))
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