import os
import logging
import asyncio
from supabase import create_client, Client
import numpy as np
from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable

class SupabaseVectorStore(VectorStoreBackend):
    """Supabase Vector Store Backend."""
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        table: str = "vectors",
        dim: int = 768,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.url = url or os.environ.get("SUPABASE_URL")
        self.api_key = api_key or os.environ.get("SUPABASE_API_KEY")
        self.table = table
        self.dim = dim
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self.client: Client = create_client(self.url, self.api_key)
        self._ensure_table()

    def _ensure_table(self):
        # Supabase/Postgres: create table if not exists (id, vector, metadata, document)
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table} (
            id TEXT PRIMARY KEY,
            vector FLOAT8[],
            metadata JSONB,
            document TEXT
        )
        """
        try:
            self.client.postgrest.rpc("execute_sql", {"sql": sql})
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
                data = {
                    "id": ids[i],
                    "vector": list(map(float, vectors[i])),
                    "metadata": metadatas[i],
                    "document": docs[i]
                }
                self.client.table(self.table).upsert(data).execute()
        await loop.run_in_executor(None, _add)
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        def _search():
            # Use Postgres L2 distance or cosine similarity if available
            sql = f"""
                SELECT id, metadata, document, vector,
                    (vector <#> %s::float8[]) AS distance
                FROM {self.table}
            """
            params = [list(map(float, query_vector))]
            if filter_criteria:
                filters = [f"metadata->>'{k}' = '{v}'" for k, v in filter_criteria.items()]
                sql += " WHERE " + " AND ".join(filters)
            sql += " ORDER BY distance ASC LIMIT %s"
            params.append(k)
            try:
                res = self.client.postgrest.rpc("execute_sql", {"sql": sql, "params": params}).execute()
                results = res.data if hasattr(res, 'data') else []
            except Exception as e:
                self.logger.error(f"Search failed: {e}")
                results = []
            search_results = []
            for row in results:
                search_results.append(SearchResult(
                    id=row.get("id"),
                    score=-row.get("distance", 0),
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
                self.client.table(self.table).delete().eq("id", id_).execute()
        await loop.run_in_executor(None, _delete)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        def _clear():
            self.client.table(self.table).delete().neq("id", "").execute()
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