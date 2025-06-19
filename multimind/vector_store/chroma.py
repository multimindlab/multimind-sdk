"""
Chroma vector store backend implementation.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
import chromadb
from chromadb.config import Settings
import asyncio

from .base import VectorStoreBackend, VectorStoreConfig, SearchResult

class ChromaBackend(VectorStoreBackend):
    """Chroma vector store backend with advanced features."""
    def __init__(
        self,
        collection_name: str = "default",
        dimension: Optional[int] = None,
        chroma_settings: Optional[Dict[str, Any]] = None,
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
        self.collection_name = collection_name
        self.dimension = dimension
        self.chroma_settings = chroma_settings or {}
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
        self.client = None
        self.collection = None

    async def initialize(self) -> None:
        """Initialize Chroma client and collection."""
        settings = Settings(**self.chroma_settings)
        self.client = chromadb.Client(settings)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"dimension": self.dimension} if self.dimension else None
        )

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to Chroma collection."""
        if not self.collection:
            await self.initialize()
        
        # Prepare documents and metadatas
        docs = [doc["content"] if isinstance(doc, dict) and "content" in doc else str(doc) for doc in documents]
        if not ids:
            ids = [f"doc_{i}" for i in range(len(docs))]
        
        # Add to collection
        self.collection.add(
            embeddings=vectors,
            documents=docs,
            metadatas=metadatas,
            ids=ids
        )
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
        """Search Chroma collection."""
        if not self.collection:
            await self.initialize()
        
        explain = explain if explain is not None else self.explain
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=filter_criteria
        )
        
        # Convert to SearchResult format
        search_results = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            doc = {"content": results["documents"][0][i]}
            score = results["distances"][0][i] if "distances" in results else 1.0
            bm25_score = None
            if self.enable_hybrid_search and query_text:
                bm25_score = self._bm25_score(query_text, doc["content"])
                score = self.hybrid_weight * score + (1 - self.hybrid_weight) * bm25_score
            if filter_criteria and not all(meta.get(k) == v for k, v in filter_criteria.items()):
                continue
            result = SearchResult(
                id=results["ids"][0][i],
                vector=query_vector,
                metadata=meta,
                document=doc,
                score=score
            )
            if explain:
                result.explanation = {
                    "vector_score": results["distances"][0][i] if "distances" in results else 1.0,
                    "bm25_score": bm25_score,
                    "final_score": score
                }
            search_results.append(result)
        
        if scoring_method and scoring_method != "weighted_sum":
            search_results = self._apply_custom_scoring(search_results, scoring_method)
        self.log_metrics('search', len(search_results))
        return search_results

    def _bm25_score(self, query_text: str, doc_text: str) -> float:
        return float(len(set(query_text.split()) & set(doc_text.split()))) / (len(doc_text.split()) + 1)

    def _apply_custom_scoring(self, results: List[SearchResult], method: str) -> List[SearchResult]:
        if method == "reciprocal_rank":
            for i, r in enumerate(results):
                r.score = 1.0 / (i + 1)
        return results

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from Chroma collection."""
        if not self.collection:
            await self.initialize()
        
        self.collection.delete(ids=ids)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self) -> None:
        """Clear Chroma collection."""
        if not self.collection:
            await self.initialize()
        
        self.collection.delete(where={})
        self.log_metrics('clear', 1)

    async def persist(self, path: str) -> None:
        """Persist Chroma collection to disk."""
        # Chroma persists automatically to the configured directory
        self.log_metrics('persist', 1)

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "ChromaBackend":
        """Load Chroma collection from disk."""
        backend = cls(**config.connection_params)
        await backend.initialize()
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