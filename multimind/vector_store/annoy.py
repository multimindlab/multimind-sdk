"""
Annoy Vector Store Backend (Pro Version)
- Async, type-safe, and extensible
- Supports hybrid search, metadata filtering, custom scoring, batch ops, persistence, monitoring, and plugin hooks
"""

from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import logging
from annoy import AnnoyIndex
import os
import asyncio

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
        **kwargs
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
        self.index = AnnoyIndex(self.vector_dim, 'angular')
        self.id_map = {}
        self.rev_id_map = {}
        self.metadata = {}
        self.documents = {}
        self.next_idx = 0
        if self.persist_path and os.path.exists(self.persist_path):
            self.index.load(self.persist_path)

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        for i, vector in enumerate(vectors):
            idx = self.next_idx
            id_str = ids[i] if ids else str(idx)
            self.index.add_item(idx, vector)
            self.id_map[id_str] = idx
            self.rev_id_map[idx] = id_str
            self.metadata[id_str] = metadatas[i]
            self.documents[id_str] = documents[i]
            self.next_idx += 1
        self.index.build(self.n_trees)
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
        explain = explain if explain is not None else self.explain
        idxs, dists = self.index.get_nns_by_vector(query_vector, k, include_distances=True)
        results = []
        for idx, dist in zip(idxs, dists):
            id_str = self.rev_id_map[idx]
            meta = self.metadata[id_str]
            doc = self.documents[id_str]
            score = 1 / (1 + dist)
            bm25_score = None
            # Hybrid search
            if self.enable_hybrid_search and query_text:
                bm25_score = self._bm25_score(query_text, doc.get("content", ""))
                score = self.hybrid_weight * score + (1 - self.hybrid_weight) * bm25_score
            # Metadata filtering
            if filter_criteria and not all(meta.get(k) == v for k, v in filter_criteria.items()):
                continue
            result = SearchResult(
                id=id_str,
                vector=query_vector,
                metadata=meta,
                document=doc,
                score=score
            )
            if explain:
                result.explanation = {
                    "vector_score": 1 / (1 + dist),
                    "bm25_score": bm25_score,
                    "final_score": score
                }
            results.append(result)
        # Custom scoring/fusion
        if scoring_method and scoring_method != "weighted_sum":
            results = self._apply_custom_scoring(results, scoring_method)
        self.log_metrics('search', len(results))
        return results

    def _bm25_score(self, query_text: str, doc_text: str) -> float:
        # Simple BM25 placeholder (replace with real BM25 if needed)
        return float(len(set(query_text.split()) & set(doc_text.split()))) / (len(doc_text.split()) + 1)

    def _apply_custom_scoring(self, results: List[SearchResult], method: str) -> List[SearchResult]:
        # Example: reciprocal rank fusion
        if method == "reciprocal_rank":
            for i, r in enumerate(results):
                r.score = 1.0 / (i + 1)
        return results

    async def delete_vectors(self, ids: List[str]) -> None:
        for id_str in ids:
            idx = self.id_map.pop(id_str, None)
            if idx is not None:
                self.rev_id_map.pop(idx, None)
                self.metadata.pop(id_str, None)
                self.documents.pop(id_str, None)
        self.index = AnnoyIndex(self.vector_dim, 'angular')
        self.next_idx = 0
        for id_str, idx in self.id_map.items():
            self.index.add_item(idx, self.documents[id_str]['vector'])
            self.next_idx += 1
        self.index.build(self.n_trees)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self) -> None:
        self.index = AnnoyIndex(self.vector_dim, 'angular')
        self.id_map.clear()
        self.rev_id_map.clear()
        self.metadata.clear()
        self.documents.clear()
        self.next_idx = 0
        self.log_metrics('clear', 1)

    async def persist(self, path: str) -> None:
        self.index.save(path)
        self.log_metrics('persist', 1)

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
        retries = self.retry_policy.get('retries', 3)
        for attempt in range(retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error: {e}, attempt {attempt+1}/{retries}")
                if attempt == retries - 1:
                    raise 