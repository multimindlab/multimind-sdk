import asyncio
from typing import Any, Callable, Dict, List, Optional

from .base import SearchResult, VectorStoreBackend

# Placeholder: Replace with actual DingoDB SDK import if available


class DingoDBBackend(VectorStoreBackend):
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
        **kwargs,
    ):
        super().__init__(
            api_key,
            endpoint,
            collection,
            enable_hybrid_search,
            hybrid_weight,
            scoring_method,
            enable_metadata_indexing,
            live_indexing,
            metrics_enabled,
            plugin_registry,
            retry_policy,
            explain,
            **kwargs,
        )
        self._store = []

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        raise NotImplementedError(
            "DingoDBBackend.add_vectors is not implemented: the DingoDB backend is a "
            "stub. Use an implemented backend such as FAISS, Chroma, Qdrant, "
            "Pinecone, Milvus, or Weaviate."
        )

    async def search(
        self,
        query_vector,
        k=5,
        query_text: Optional[str] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        scoring_method: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None,
        explain: Optional[bool] = None,
    ) -> List[SearchResult]:
        raise NotImplementedError(
            "DingoDBBackend.search is not implemented: the DingoDB backend is a stub. "
            "Use an implemented backend such as FAISS, Chroma, Qdrant, Pinecone, "
            "Milvus, or Weaviate."
        )

    def _bm25_score(self, query_text: str, doc_text: str) -> float:
        return float(len(set(query_text.split()) & set(doc_text.split()))) / (
            len(doc_text.split()) + 1
        )

    def _apply_custom_scoring(self, results: List[SearchResult], method: str) -> List[SearchResult]:
        if method == "reciprocal_rank":
            for i, r in enumerate(results):
                r.score = 1.0 / (i + 1)
        return results

    async def delete_vectors(self, ids):
        raise NotImplementedError(
            "DingoDBBackend.delete_vectors is not implemented: the DingoDB backend is "
            "a stub. Use an implemented backend such as FAISS, Chroma, Qdrant, "
            "Pinecone, Milvus, or Weaviate."
        )

    async def clear(self):
        raise NotImplementedError(
            "DingoDBBackend.clear is not implemented: the DingoDB backend is a stub. "
            "Use an implemented backend such as FAISS, Chroma, Qdrant, Pinecone, "
            "Milvus, or Weaviate."
        )

    async def persist(self, path):
        raise NotImplementedError(
            "DingoDBBackend.persist is not implemented: the DingoDB backend is a "
            "stub. Use an implemented backend such as FAISS, Chroma, Qdrant, "
            "Pinecone, Milvus, or Weaviate."
        )

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
        retries = self.retry_policy.get("retries", 3)
        for attempt in range(retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error: {e}, attempt {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    raise

    def add(self, vector, metadata=None):
        raise NotImplementedError(
            "DingoDBBackend.add is not implemented: the DingoDB backend is a stub. "
            "Use an implemented backend such as FAISS, Chroma, Qdrant, Pinecone, "
            "Milvus, or Weaviate."
        )

    def search(self, query_vector, top_k=3):
        raise NotImplementedError(
            "DingoDBBackend.search is not implemented: the DingoDB backend is a stub. "
            "Use an implemented backend such as FAISS, Chroma, Qdrant, Pinecone, "
            "Milvus, or Weaviate."
        )

    def delete(self, index):
        raise NotImplementedError(
            "DingoDBBackend.delete is not implemented: the DingoDB backend is a stub. "
            "Use an implemented backend such as FAISS, Chroma, Qdrant, Pinecone, "
            "Milvus, or Weaviate."
        )
