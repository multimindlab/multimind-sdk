from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
try:
    from cassandra.cluster import Cluster
    from cassandra.auth import PlainTextAuthProvider
except ImportError:
    Cluster = None
    PlainTextAuthProvider = None

class CassandraBackend(VectorStoreBackend):
    def __init__(
        self,
        contact_points: Optional[List[str]] = None,
        port: int = 9042,
        username: Optional[str] = None,
        password: Optional[str] = None,
        keyspace: str = "vector_keyspace",
        table: str = "vectors",
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
        self.contact_points = contact_points or os.environ.get("CASSANDRA_CONTACT_POINTS", "127.0.0.1").split(",")
        self.port = port
        self.username = username or os.environ.get("CASSANDRA_USERNAME")
        self.password = password or os.environ.get("CASSANDRA_PASSWORD")
        self.keyspace = keyspace
        self.table = table
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
        if Cluster is None or PlainTextAuthProvider is None:
            raise ImportError("cassandra-driver is not installed. Please install it to use this backend.")
        auth_provider = None
        if self.username and self.password:
            auth_provider = PlainTextAuthProvider(username=self.username, password=self.password)
        self.cluster = Cluster(contact_points=self.contact_points, port=self.port, auth_provider=auth_provider)
        self.session = self.cluster.connect(self.keyspace)

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        for i, vector in enumerate(vectors):
            doc_id = ids[i] if ids else None
            # Placeholder: actual vector storage logic depends on schema
            self.session.execute(
                f"INSERT INTO {self.table} (id, vector, metadata, document) VALUES (%s, %s, %s, %s)",
                (doc_id, vector, metadatas[i], documents[i])
            )
        if self.live_indexing:
            await self._run_plugin('on_live_index', vectors, metadatas, documents, ids)
        self.log_metrics('add_vectors', len(vectors))

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        explain = explain if explain is not None else self.explain
        # Placeholder: Cassandra does not natively support vector search; implement custom logic or use an extension
        results = []
        # Implement vector search logic here
        self.log_metrics('search', len(results))
        return results

    def _bm25_score(self, query_text: str, doc_text: str) -> float:
        return float(len(set(query_text.split()) & set(doc_text.split()))) / (len(doc_text.split()) + 1)

    def _apply_custom_scoring(self, results: List[SearchResult], method: str) -> List[SearchResult]:
        if method == "reciprocal_rank":
            for i, r in enumerate(results):
                r.score = 1.0 / (i + 1)
        return results

    async def delete_vectors(self, ids):
        for doc_id in ids:
            self.session.execute(f"DELETE FROM {self.table} WHERE id = %s", (doc_id,))
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        self.session.execute(f"TRUNCATE {self.table}")
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