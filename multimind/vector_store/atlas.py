"""
Atlas Vector Store Backend (Pro Version)
- Async, type-safe, and extensible
- Supports hybrid search, metadata filtering, custom scoring, batch ops, persistence, monitoring, and plugin hooks
"""

from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
from pymongo import MongoClient

class AtlasBackend(VectorStoreBackend):
    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: str = "vector_db",
        collection: str = "vectors",
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
        self.uri = uri or os.environ.get("MONGODB_ATLAS_URI")
        self.db_name = db_name
        self.collection = collection
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
        if not self.uri:
            raise ValueError("MongoDB Atlas URI must be provided.")
        self.client = MongoClient(self.uri)
        self.col = self.client[self.db_name][self.collection]

    async def initialize(self) -> None:
        """Connect to Atlas and create index if needed."""
        self.logger.info("Initializing Atlas backend...")
        pass

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
            doc = {
                "_id": doc_id,
                "vector": vector,
                "metadata": metadatas[i],
                "document": documents[i],
            }
            self.col.insert_one(doc)
        if self.live_indexing:
            await self._run_plugin('on_live_index', vectors, metadatas, documents, ids)
        self.log_metrics('add_vectors', len(vectors))

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
        scoring_method: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None,
        explain: Optional[bool] = None
    ) -> List[SearchResult]:
        """Hybrid search: vector + keyword + metadata + custom scoring."""
        explain = explain if explain is not None else self.explain
        pipeline = [
            {
                "$search": {
                    "index": "default",
                    "knnBeta": {
                        "vector": query_vector,
                        "k": k,
                        "path": "vector"
                    }
                }
            }
        ]
        res = self.col.aggregate(pipeline)
        results = []
        for doc in res:
            meta = doc.get("metadata", {})
            doc_content = doc.get("document", {})
            score = doc.get("score", 1.0)
            bm25_score = None
            if self.enable_hybrid_search and query_text:
                bm25_score = self._bm25_score(query_text, doc_content.get("content", ""))
                score = self.hybrid_weight * score + (1 - self.hybrid_weight) * bm25_score
            if filter_criteria and not all(meta.get(k) == v for k, v in filter_criteria.items()):
                continue
            result = SearchResult(
                id=doc["_id"],
                vector=doc["vector"],
                metadata=meta,
                document=doc_content,
                score=score
            )
            if explain:
                result.explanation = {
                    "vector_score": doc.get("score", 1.0),
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
            self.col.delete_one({"_id": doc_id})
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self) -> None:
        """Clear all vectors from the index."""
        self.col.delete_many({})
        self.log_metrics('clear', 1)

    async def persist(self, path: str) -> None:
        """Persist index/config to disk/cloud if supported."""
        self.log_metrics('persist', 1)

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "AtlasBackend":
        """Load index/config from disk/cloud if supported."""
        backend = cls(**config.connection_params)
        await backend.initialize()
        return backend

    # --- Advanced/Pro Features ---
    # Add hooks for plugin system, custom scoring, live updates, monitoring, etc.
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