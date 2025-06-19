from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
try:
    from elasticsearch import Elasticsearch
except ImportError:
    Elasticsearch = None

class ElasticsearchBackend(VectorStoreBackend):
    def __init__(
        self,
        hosts: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        index_name: str = "vectors",
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
        self.hosts = hosts or os.environ.get("ELASTICSEARCH_HOSTS", "localhost:9200").split(",")
        self.api_key = api_key or os.environ.get("ELASTICSEARCH_API_KEY")
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
        if Elasticsearch is None:
            raise ImportError("elasticsearch is not installed. Please install it to use this backend.")
        self.client = Elasticsearch(self.hosts, api_key=self.api_key)

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
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

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
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

    async def delete_vectors(self, ids):
        for doc_id in ids:
            self.client.delete(index=self.index_name, id=doc_id)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        self.client.indices.delete(index=self.index_name, ignore=[400, 404])
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