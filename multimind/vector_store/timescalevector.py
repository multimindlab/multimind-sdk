import os
import logging
import asyncio
import numpy as np
import asyncpg
import psycopg2
from timescale_vector import TimescaleVector
from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable

class TimescaleVectorStore(VectorStoreBackend):
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "test_db",
        table: str = "vectors",
        dim: int = 768,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.host = host or os.environ.get("TIMESCALE_HOST", "localhost")
        self.port = port or int(os.environ.get("TIMESCALE_PORT", 5432))
        self.user = user or os.environ.get("TIMESCALE_USER", "postgres")
        self.password = password or os.environ.get("TIMESCALE_PASSWORD", "")
        self.database = database
        self.table = table
        self.dim = dim
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self.conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.database
        )
        self._ensure_table()
        self.ts_vector = TimescaleVector(self.conn)

    def _ensure_table(self):
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id TEXT PRIMARY KEY,
                    vector VECTOR({self.dim}),
                    metadata JSONB,
                    document TEXT
                )
            """)
            self.conn.commit()

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        loop = asyncio.get_event_loop()
        def _add():
            with self.conn.cursor() as cur:
                for i in range(n):
                    cur.execute(f"""
                        INSERT INTO {self.table} (id, vector, metadata, document)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET vector = EXCLUDED.vector, metadata = EXCLUDED.metadata, document = EXCLUDED.document
                    """, (ids[i], list(map(float, vectors[i])), metadatas[i], docs[i]))
                self.conn.commit()
        await loop.run_in_executor(None, _add)
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        def _search():
            with self.conn.cursor() as cur:
                where = []
                params = []
                if filter_criteria:
                    for kf, vf in filter_criteria.items():
                        where.append(f"metadata->>'{kf}' = %s")
                        params.append(vf)
                where_clause = f"WHERE {' AND '.join(where)}" if where else ""
                sql = f"""
                    SELECT id, metadata, document, vector <#> %s AS distance
                    FROM {self.table}
                    {where_clause}
                    ORDER BY distance ASC
                    LIMIT %s
                """
                params = [list(map(float, query_vector))] + params + [k]
                cur.execute(sql, params)
                results = cur.fetchall()
                search_results = []
                for row in results:
                    id_, meta, doc, dist = row
                    search_results.append(SearchResult(
                        id=id_,
                        score=-dist,
                        metadata=meta,
                        document=doc
                    ))
                return search_results
        search_results = await loop.run_in_executor(None, _search)
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        def _delete():
            with self.conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self.table} WHERE id IN ({','.join(['%s']*len(ids))})", ids)
                self.conn.commit()
        await loop.run_in_executor(None, _delete)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        def _clear():
            with self.conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.table}")
                self.conn.commit()
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