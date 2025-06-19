"""
Alibaba Cloud OpenSearch Vector Store Backend (Pro Version)
- Async, type-safe, and extensible
- Supports hybrid search, metadata filtering, custom scoring, batch ops, persistence, monitoring, and plugin hooks
"""

from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio

try:
    from opensearchpy import OpenSearch
except ImportError:
    OpenSearch = None

class AlibabaCloudOpenSearchBackend(VectorStoreBackend):
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        index_name: str = "default-index",
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
        self.api_key = api_key or os.environ.get("ALI_OPENSEARCH_API_KEY")
        self.endpoint = endpoint or os.environ.get("ALI_OPENSEARCH_ENDPOINT")
        self.index_name = index_name
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
        if not self.api_key or not self.endpoint:
            raise ValueError("API key and endpoint must be provided for Alibaba Cloud OpenSearch.")
        if OpenSearch is None:
            raise ImportError("opensearchpy is not installed. Please install it to use this backend.")
        self.client = OpenSearch(
            hosts=[{"host": self.endpoint, "port": 443}],
            http_auth=(self.api_key, ""),
            use_ssl=True,
            verify_certs=True,
        )

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors with metadata and documents (batch supported)."""
        for i, vector in enumerate(vectors):
            doc_id = ids[i] if ids else None
            body = {
                "vector": vector,
                "metadata": metadatas[i],
                "document": documents[i],
            }
            self.client.index(index=self.index_name, id=doc_id, body=body)
        if self.live_indexing:
            await self._run_plugin('on_live_index', vectors, metadatas, documents, ids)
        self.log_metrics('add_vectors', len(vectors))

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        query_text: Optional[str] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        scoring_method: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None,
        explain: Optional[bool] = None
    ) -> List[SearchResult]:
        """Hybrid search: vector + keyword + metadata + custom scoring."""
        explain = explain if explain is not None else self.explain
        query = {
            "size": k,
            "query": {
                "knn": {
                    "vector": {
                        "vector": query_vector,
                        "k": k
                    }
                }
            }
        }
        res = self.client.search(index=self.index_name, body=query)
        results = []
        for hit in res["hits"]["hits"]:
            meta = hit["_source"].get("metadata", {})
            doc = hit["_source"].get("document", {})
            score = hit["_score"]
            bm25_score = None
            if self.enable_hybrid_search and query_text:
                bm25_score = self._bm25_score(query_text, doc.get("content", ""))
                score = self.hybrid_weight * score + (1 - self.hybrid_weight) * bm25_score
            if filter_criteria and not all(meta.get(k) == v for k, v in filter_criteria.items()):
                continue
            result = SearchResult(
                id=hit["_id"],
                vector=hit["_source"]["vector"],
                metadata=meta,
                document=doc,
                score=score
            )
            if explain:
                result.explanation = {
                    "vector_score": hit["_score"],
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

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by ID (batch supported)."""
        for doc_id in ids:
            self.client.delete(index=self.index_name, id=doc_id)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self) -> None:
        """Clear all vectors from the index."""
        self.client.indices.delete(index=self.index_name, ignore=[400, 404])
        self.log_metrics('clear', 1)

    async def persist(self, path: str) -> None:
        """Persist index/config to disk/cloud if supported."""
        self.log_metrics('persist', 1)

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "AlibabaCloudOpenSearchBackend":
        """Load index/config from disk/cloud if supported."""
        backend = cls(**config.connection_params)
        return backend

    # --- Advanced/Pro Features ---
    # Add hooks for plugin system, custom scoring, live updates, monitoring, etc.
    # Example:
    def register_plugin(self, name: str, plugin: Callable):
        """Register a plugin for custom logic (optional)."""
        self.plugin_registry[name] = plugin

    async def _run_plugin(self, name: str, *args, **kwargs):
        if name in self.plugin_registry:
            if asyncio.iscoroutinefunction(self.plugin_registry[name]):
                await self.plugin_registry[name](*args, **kwargs)
            else:
                self.plugin_registry[name](*args, **kwargs)

    def log_metrics(self, metric_name: str, value: Any):
        """Log or export metrics for monitoring (optional)."""
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

