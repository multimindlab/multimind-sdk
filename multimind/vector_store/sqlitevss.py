from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
import sqlite3
import numpy as np

class SQLiteVSSBackend(VectorStoreBackend):
    def __init__(
        self,
        db_path: str = "vectors.db",
        table: str = "vectors",
        dim: int = 768,
        vss_extension_path: Optional[str] = None,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.db_path = db_path
        self.table = table
        self.dim = dim
        self.vss_extension_path = vss_extension_path or os.environ.get("SQLITE_VSS_EXTENSION_PATH")
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        if self.vss_extension_path:
            self._conn.enable_load_extension(True)
            self._conn.load_extension(self.vss_extension_path)
        self._ensure_table()

    def _ensure_table(self):
        with self._conn:
            self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id TEXT PRIMARY KEY,
                    vector BLOB,
                    metadata TEXT,
                    document TEXT
                )
            """)
            # Create VSS index if not exists (user must have loaded the extension)
            try:
                self._conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.table}_vss USING vss0(vector({self.dim}))")
            except sqlite3.OperationalError:
                pass  # Extension not loaded or already exists

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        loop = asyncio.get_event_loop()
        def _add():
            with self._conn:
                for i in range(n):
                    self._conn.execute(f"""
                        INSERT OR REPLACE INTO {self.table} (id, vector, metadata, document)
                        VALUES (?, ?, ?, ?)
                    """, (ids[i], np.array(vectors[i], dtype=np.float32).tobytes(), str(metadatas[i]), docs[i]))
                    # Insert into VSS index
                    try:
                        self._conn.execute(f"INSERT OR REPLACE INTO {self.table}_vss(rowid, vector) VALUES ((SELECT rowid FROM {self.table} WHERE id = ?), ?)", (ids[i], np.array(vectors[i], dtype=np.float32).tobytes()))
                    except sqlite3.OperationalError:
                        pass  # VSS extension not loaded
        await loop.run_in_executor(None, _add)
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        def _search():
            try:
                sql = f"""
                    SELECT t.id, t.metadata, t.document, v.distance
                    FROM {self.table}_vss v
                    JOIN {self.table} t ON v.rowid = t.rowid
                    WHERE v.vector MATCH ?
                    ORDER BY v.distance ASC
                    LIMIT ?
                """
                params = [np.array(query_vector, dtype=np.float32).tobytes(), k]
                cur = self._conn.execute(sql, params)
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
            except sqlite3.OperationalError:
                return []  # VSS extension not loaded
        search_results = await loop.run_in_executor(None, _search)
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        def _delete():
            with self._conn:
                for id_ in ids:
                    self._conn.execute(f"DELETE FROM {self.table} WHERE id = ?", (id_,))
                    try:
                        self._conn.execute(f"DELETE FROM {self.table}_vss WHERE rowid = (SELECT rowid FROM {self.table} WHERE id = ?)", (id_,))
                    except sqlite3.OperationalError:
                        pass
        await loop.run_in_executor(None, _delete)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        def _clear():
            with self._conn:
                self._conn.execute(f"DELETE FROM {self.table}")
                try:
                    self._conn.execute(f"DELETE FROM {self.table}_vss")
                except sqlite3.OperationalError:
                    pass
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