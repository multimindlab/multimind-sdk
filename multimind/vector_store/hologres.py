from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
from hologres_vector import HologresVector

class HologresBackend(VectorStoreBackend):
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        table: str = "vectors",
        vector_dim: Optional[int] = None,
        table_schema: Optional[Dict[str, str]] = None,
        distance_method: str = "SquaredEuclidean",
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
        self.host = host or os.environ.get("HOLO_HOST")
        self.port = port or os.environ.get("HOLO_PORT")
        self.dbname = dbname or os.environ.get("HOLO_DBNAME")
        self.user = user or os.environ.get("HOLO_USER")
        self.password = password or os.environ.get("HOLO_PASSWORD")
        self.table = table
        self.vector_dim = vector_dim
        self.table_schema = table_schema or {}
        self.distance_method = distance_method
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
        if not all([self.host, self.port, self.dbname, self.user, self.password, self.vector_dim]):
            raise ValueError("Hologres connection parameters and vector_dim must be provided.")
        conn_str = HologresVector.connection_string_from_db_params(
            self.host, self.port, self.dbname, self.user, self.password
        )
        self.client = HologresVector(
            conn_str,
            self.vector_dim,
            table_name=self.table,
            table_schema=self.table_schema,
            distance_method=self.distance_method,
            pre_delete_table=False,
        )

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        # Hologres expects vectors, ids, schema_datas, metadatas
        schema_datas = documents if documents else [{} for _ in vectors]
        metadatas = metadatas if metadatas else [{} for _ in vectors]
        ids = ids if ids else [str(i) for i in range(len(vectors))]
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.client.upsert_vectors(vectors, ids, schema_datas=schema_datas, metadatas=metadatas)
        )
        if self.live_indexing:
            await self._run_plugin('on_live_index', vectors, metadatas, documents, ids)
        self.log_metrics('add_vectors', len(vectors))

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        explain = explain if explain is not None else self.explain
        # Integrated query: nearest neighbor + filter
        search_kwargs = {"k": k}
        if filter_criteria:
            search_kwargs["schema_data_filters"] = filter_criteria
        res = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.client.search(query_vector, **search_kwargs)
        )
        results = []
        for doc in res:
            meta = doc.get("metadata", {})
            doc_content = {k: v for k, v in doc.items() if k not in ["id", "vector", "metadata", "distance"]}
            score = 1.0 / (1.0 + doc.get("distance", 0.0))
            bm25_score = None
            if self.enable_hybrid_search and query_text:
                bm25_score = self._bm25_score(query_text, str(doc_content))
                score = self.hybrid_weight * score + (1 - self.hybrid_weight) * bm25_score
            if filter_criteria and not all(doc_content.get(k) == v for k, v in filter_criteria.items()):
                continue
            result = SearchResult(
                id=doc.get("id"),
                vector=doc.get("vector"),
                metadata=meta,
                document=doc_content,
                score=score
            )
            if explain:
                result.explanation = {
                    "distance": doc.get("distance", 0.0),
                    "bm25_score": bm25_score,
                    "final_score": score
                }
            results.append(result)
        if scoring_method and scoring_method != "weighted_sum":
            results = self._apply_custom_scoring(results, scoring_method)
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
        # Delete by id using schema_data_filters
        for doc_id in ids:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.delete_vectors(schema_data_filters={"id": doc_id})
            )
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        # Delete all data
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.client.delete_vectors()
        )
        self.log_metrics('clear', 1)

    async def persist(self, path):
        # Hologres is a managed service, so persistence is not typically needed
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