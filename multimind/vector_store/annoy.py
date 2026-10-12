"""
Annoy Vector Store Backend (Pro Version)
- Async, type-safe, and extensible
- Supports hybrid search, metadata filtering, custom scoring, batch ops, persistence, monitoring, and plugin hooks
"""

import asyncio
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from annoy import AnnoyIndex

from .base import SearchResult, VectorStoreBackend, VectorStoreConfig


class AnnoyBackend(VectorStoreBackend):
    def __init__(
        self,
        vector_dim: int,
        n_trees: int = 10,
        persist_path: Optional[str] = None,
        enable_hybrid_search: bool = False,
        hybrid_weight: float = 0.5,
        scoring_method: str = "weighted_sum",
        enable_metadata_indexing: bool = False,
        live_indexing: bool = False,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        explain: bool = False,
        **kwargs,
    ):
        self.vector_dim = vector_dim
        self.n_trees = n_trees
        self.persist_path = persist_path
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
        self.index = AnnoyIndex(self.vector_dim, "angular")
        self.id_map = {}
        self.rev_id_map = {}
        self.metadata = {}
        self.documents = {}
        # Raw vectors by id: Annoy indexes are immutable once built, so adds
        # and deletes rebuild the index from these.
        self._vectors: Dict[str, List[float]] = {}
        # Monotonic counter for auto-generated ids only; kept separate from
        # the Annoy item indices (see _rebuild_index) which must stay dense.
        self._auto_id_counter = 0
        if self.persist_path and os.path.exists(self.persist_path):
            self.index.load(self.persist_path)

    async def initialize(self) -> None:
        """No-op: the Annoy index is set up eagerly in __init__."""
        pass

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        for i, vector in enumerate(vectors):
            if ids:
                id_str = ids[i]
            else:
                id_str = str(self._auto_id_counter)
                self._auto_id_counter += 1
            self.metadata[id_str] = metadatas[i]
            self.documents[id_str] = documents[i]
            self._vectors[id_str] = list(vector)
        self._rebuild_index()
        if self.live_indexing:
            await self._run_plugin("on_live_index", vectors, metadatas, documents, ids)
        self.log_metrics("add_vectors", len(vectors))

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
        scoring_method: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None,
        explain: Optional[bool] = None,
    ) -> List[SearchResult]:
        explain = explain if explain is not None else self.explain
        idxs, dists = self.index.get_nns_by_vector(query_vector, k, include_distances=True)
        results = []
        for idx, dist in zip(idxs, dists):
            id_str = self.rev_id_map[idx]
            meta = self.metadata[id_str]
            doc = self.documents[id_str]
            score = 1 / (1 + dist)
            keyword_score = None
            # Hybrid search
            if self.enable_hybrid_search and query_text:
                keyword_score = self._token_overlap_score(query_text, doc.get("content", ""))
                score = self.hybrid_weight * score + (1 - self.hybrid_weight) * keyword_score
            # Metadata filtering
            if filter_criteria and not all(meta.get(k) == v for k, v in filter_criteria.items()):
                continue
            result = SearchResult(
                id=id_str, vector=query_vector, metadata=meta, document=doc, score=score
            )
            if explain:
                result.explanation = {
                    "vector_score": 1 / (1 + dist),
                    "keyword_score": keyword_score,
                    "final_score": score,
                }
            results.append(result)
        # Custom scoring/fusion
        if scoring_method and scoring_method != "weighted_sum":
            results = self._apply_custom_scoring(results, scoring_method)
        self.log_metrics("search", len(results))
        return results

    def _token_overlap_score(self, query_text: str, doc_text: str) -> float:
        # Naive token-overlap keyword score; not BM25
        return float(len(set(query_text.split()) & set(doc_text.split()))) / (
            len(doc_text.split()) + 1
        )

    def _apply_custom_scoring(self, results: List[SearchResult], method: str) -> List[SearchResult]:
        # Example: reciprocal rank fusion
        if method == "reciprocal_rank":
            for i, r in enumerate(results):
                r.score = 1.0 / (i + 1)
        return results

    def _rebuild_index(self) -> None:
        """Rebuild the (immutable-once-built) Annoy index from stored vectors.

        Renumbers every id to a contiguous 0..n-1 range: Annoy silently
        returns wrong/incomplete results when item indices have gaps (e.g.
        after a delete), so ids are kept dense across every add/delete.
        """
        self.index = AnnoyIndex(self.vector_dim, "angular")
        self.id_map = {}
        self.rev_id_map = {}
        for new_idx, id_str in enumerate(self._vectors):
            self.index.add_item(new_idx, self._vectors[id_str])
            self.id_map[id_str] = new_idx
            self.rev_id_map[new_idx] = id_str
        self.index.build(self.n_trees)

    async def delete_vectors(self, ids: List[str]) -> None:
        for id_str in ids:
            if id_str in self._vectors:
                self.metadata.pop(id_str, None)
                self.documents.pop(id_str, None)
                self._vectors.pop(id_str, None)
        self._rebuild_index()
        self.log_metrics("delete_vectors", len(ids))

    async def clear(self) -> None:
        self.index = AnnoyIndex(self.vector_dim, "angular")
        self.id_map.clear()
        self.rev_id_map.clear()
        self.metadata.clear()
        self.documents.clear()
        self._vectors.clear()
        self._auto_id_counter = 0
        self.log_metrics("clear", 1)

    async def persist(self, path: str) -> None:
        self.index.save(path)
        self.log_metrics("persist", 1)

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "AnnoyBackend":
        backend = cls(**config.connection_params)
        backend.index.load(path)
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
        retries = self.retry_policy.get("retries", 3)
        for attempt in range(retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error: {e}, attempt {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    raise
