import asyncio
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from .base import SearchResult, VectorStoreBackend

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
except ImportError:
    SearchClient = None
    AzureKeyCredential = None


class AzureSearchBackend(VectorStoreBackend):
    def __init__(
        self,
        endpoint: Optional[str] = None,
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
        **kwargs,
    ):
        self.endpoint = endpoint or os.environ.get("AZURE_SEARCH_ENDPOINT")
        self.api_key = api_key or os.environ.get("AZURE_SEARCH_API_KEY")
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
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure Search endpoint and API key must be provided.")
        if SearchClient is None or AzureKeyCredential is None:
            raise ImportError(
                "azure-search-documents is not installed. Please install it to use this backend."
            )
        self.client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.api_key),
        )

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        actions = []
        for i, vector in enumerate(vectors):
            doc_id = ids[i] if ids else None
            doc = {
                "id": doc_id,
                "vector": vector,
                "metadata": metadatas[i],
                "document": documents[i],
            }
            actions.append({"@search.action": "upload", **doc})
        self.client.upload_documents(documents=actions)
        if self.live_indexing:
            await self._run_plugin("on_live_index", vectors, metadatas, documents, ids)
        self.log_metrics("add_vectors", len(vectors))

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
            "AzureSearchBackend.search is not implemented: the previous "
            "implementation ignored the query vector and ran a text-only search. Use "
            "an implemented backend such as FAISS, Chroma, Qdrant, Pinecone, Milvus, "
            "or Weaviate."
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
        actions = [{"@search.action": "delete", "id": doc_id} for doc_id in ids]
        self.client.upload_documents(documents=actions)
        self.log_metrics("delete_vectors", len(ids))

    async def clear(self):
        # Azure Search does not have a direct clear; delete all docs by query
        # Placeholder: implement as needed
        raise NotImplementedError(
            "AzureSearchBackend.clear is not implemented. Delete documents via "
            "delete_vectors or recreate the index instead."
        )

    async def persist(self, path):
        self.log_metrics("persist", 1)

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
