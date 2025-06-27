from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
import lancedb
import numpy as np
import pyarrow as pa

class LanceDBBackend(VectorStoreBackend):
    def __init__(
        self,
        uri: Optional[str] = None,
        table: str = "vectors",
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        vector_dim: Optional[int] = None,
        enable_hybrid_search: bool = False,
        hybrid_weight: float = 0.5,
        scoring_method: str = "weighted_sum",
        enable_metadata_indexing: bool = False,
        live_indexing: bool = False,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        explain: bool = False,
        **kwargs
    ):
        self.uri = uri or os.environ.get("LANCEDB_URI", "./lancedb_data")
        self.table = table
        self.api_key = api_key or os.environ.get("LANCEDB_API_KEY")
        self.region = region or os.environ.get("LANCEDB_REGION")
        self.vector_dim = vector_dim
        self.enable_hybrid_search = enable_hybrid_search
        self.hybrid_weight = hybrid_weight
        self.scoring_method = scoring_method
        self.enable_metadata_indexing = enable_metadata_indexing
        self.live_indexing = live_indexing
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.explain = explain
        self.logger = logging.getLogger(__name__)
        self._db = None
        self._tbl = None
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _initialize(self):
        async with self._init_lock:
            if self._initialized:
                return
            # LanceDB supports both local and cloud
            if self.api_key and self.region:
                self._db = await lancedb.connect_async(
                    uri=self.uri, api_key=self.api_key, region=self.region
                )
            else:
                self._db = await lancedb.connect_async(self.uri)
            # Try to open table, or create if not exists
            try:
                self._tbl = await self._db.open_table(self.table)
            except Exception:
                # Create table with default schema if not exists
                if not self.vector_dim:
                    raise ValueError("vector_dim must be provided to create a new table.")
                schema = pa.schema([
                    pa.field("id", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), self.vector_dim)),
                    pa.field("metadata", pa.struct([])),
                    pa.field("document", pa.string()),
                ])
                self._tbl = await self._db.create_table(self.table, schema=schema)
            self._initialized = True

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        await self._initialize()
        data = []
        for i, vector in enumerate(vectors):
            entry = {
                "id": ids[i] if ids else str(i),
                "vector": vector,
                "metadata": metadatas[i] if metadatas else {},
                "document": documents[i] if documents else "",
            }
            data.append(entry)
        await self._tbl.add(data)
        if self.live_indexing:
            await self._run_plugin('on_live_index', vectors, metadatas, documents, ids)
        self.log_metrics('add_vectors', len(vectors))

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        await self._initialize()
        scoring_method = scoring_method or self.scoring_method
        explain = explain if explain is not None else self.explain
        # LanceDB async search
        search_query = self._tbl.search(query_vector)
        if filter_criteria:
            # Convert filter_criteria dict to SQL WHERE string
            where_clause = self._dict_to_sql_where(filter_criteria)
            search_query = search_query.where(where_clause)
        if scoring_method in ("cosine", "dot", "l2"):
            search_query = search_query.distance_type(scoring_method)
        results = await (await search_query.limit(k)).to_list()
        search_results = [
            SearchResult(
                id=row.get("id"),
                score=row.get("_distance", 0.0),
                metadata=row.get("metadata", {}),
                document=row.get("document", "")
            ) for row in results
        ]
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        await self._initialize()
        await self._tbl.delete(ids=ids)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        await self._initialize()
        await self._tbl.delete(delete_all=True)
        self.log_metrics('clear', 1)

    async def persist(self, path):
        # LanceDB is persistent by default; this is a no-op
        self.log_metrics('persist', 1)

    @classmethod
    async def load(cls, path, config):
        backend = cls(**config.connection_params)
        await backend._initialize()
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

    def _dict_to_sql_where(self, filter_criteria: Dict[str, Any]) -> str:
        # Simple AND-joined SQL WHERE clause from dict
        clauses = []
        for k, v in filter_criteria.items():
            if isinstance(v, str):
                clauses.append(f"{k} = '{v}'")
            else:
                clauses.append(f"{k} = {v}")
        return " AND ".join(clauses) 