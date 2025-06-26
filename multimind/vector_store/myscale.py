from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
import clickhouse_connect
import numpy as np

class MyScaleBackend(VectorStoreBackend):
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "default",
        table: str = "vectors",
        dim: int = 768,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.host = host or os.environ.get("MYSCALE_HOST", "localhost")
        self.port = port or int(os.environ.get("MYSCALE_PORT", 9000))
        self.user = user or os.environ.get("MYSCALE_USER", "default")
        self.password = password or os.environ.get("MYSCALE_PASSWORD", "")
        self.database = database
        self.table = table
        self.dim = dim
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self._client = clickhouse_connect.get_client(
            host=self.host,
            port=self.port,
            username=self.user,
            password=self.password,
            database=self.database
        )
        # Ensure table exists
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table} (
            id String,
            vector Array(Float32),
            metadata String,
            document String
        ) ENGINE = MergeTree() ORDER BY id
        """
        self._client.command(create_table_sql)

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        rows = [
            [ids[i], list(map(float, vectors[i])), str(metadatas[i]), docs[i]]
            for i in range(n)
        ]
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._client.insert(self.table, rows, column_names=["id", "vector", "metadata", "document"]))
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        # MyScale supports vector search using cosineDistance or L2Distance
        # We'll use cosine similarity by default
        query_vec = np.array(query_vector, dtype=np.float32)
        filter_sql = ""
        if filter_criteria:
            filter_clauses = [f"JSONExtractString(metadata, '{k}') = '{v}'" for k, v in filter_criteria.items()]
            filter_sql = " AND ".join(filter_clauses)
        where_clause = f"WHERE {filter_sql}" if filter_sql else ""
        sql = f"""
        SELECT id, vector, metadata, document,
            1 - cosineDistance(vector, {list(query_vec)}) AS score
        FROM {self.table}
        {where_clause}
        ORDER BY score DESC
        LIMIT {k}
        """
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: self._client.query(sql).result_rows)
        search_results = [
            SearchResult(
                id=row[0],
                score=row[4],
                metadata=row[2],
                document=row[3]
            ) for row in results
        ]
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        ids_list = ",".join([f"'" + str(i) + "'" for i in ids])
        sql = f"DELETE FROM {self.table} WHERE id IN ({ids_list})"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._client.command(sql))
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        sql = f"TRUNCATE TABLE {self.table}"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._client.command(sql))
        self.log_metrics('clear', 1)

    async def persist(self, path):
        # MyScale is persistent by default
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