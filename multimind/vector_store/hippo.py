from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
from hippo_api import HippoClient

class HippoBackend(VectorStoreBackend):
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
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
        self.api_key = api_key or os.environ.get("HIPPO_API_KEY")
        self.endpoint = endpoint or os.environ.get("HIPPO_ENDPOINT")
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
        if not self.api_key or not self.endpoint:
            raise ValueError("Hippo API key and endpoint must be provided.")
        self.client = HippoClient(api_key=self.api_key, endpoint=self.endpoint)
        self.col = self.client.collection(self.collection)

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        # Hippo expects a list of dicts with 'vector', 'metadata', 'document', and optional 'id'
        items = []
        for i, vector in enumerate(vectors):
            item = {
                "vector": vector,
                "metadata": metadatas[i] if metadatas else {},
                "document": documents[i] if documents else {},
            }
            if ids:
                item["id"] = ids[i]
            items.append(item)
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.col.insert_many(items)
        )
        if self.live_indexing:
            await self._run_plugin('on_live_index', vectors, metadatas, documents, ids)
        self.log_metrics('add_vectors', len(vectors))

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        explain = explain if explain is not None else self.explain
        # Build search query
        query = {"vector": query_vector, "k": k}
        if filter_criteria:
            query["filter"] = filter_criteria
        res = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.col.search(query)
        )
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
                id=doc.get("id"),
                vector=doc.get("vector"),
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

    async def delete_vectors(self, ids):
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: [self.col.delete_one({"id": doc_id}) for doc_id in ids]
        )
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.col.delete_many({})
        )
        self.log_metrics('clear', 1)

    async def persist(self, path):
        # Hippo is a managed service, so persistence is not typically needed
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