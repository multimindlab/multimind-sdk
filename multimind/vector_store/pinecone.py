"""
Pinecone vector store backend implementation.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
import os
import asyncio
import pinecone

from .base import VectorStoreBackend, VectorStoreConfig, SearchResult

class PineconeBackend(VectorStoreBackend):
    """Production-grade Pinecone vector store backend."""
    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: str = "documents",
        dimension: int = 768,
        metric: str = "cosine",
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self.environment = environment or os.environ.get("PINECONE_ENVIRONMENT")
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self.index = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        pinecone.init(api_key=self.api_key, environment=self.environment)
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric
            )
        self.index = pinecone.Index(self.index_name)
        self._initialized = True

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to Pinecone."""
        await self.initialize()
        n = len(vectors)
        ids = ids or [f"doc_{i}" for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        # Prepare upsert tuples
        upsert_data = []
        for i in range(n):
            meta = metadatas[i].copy()
            if isinstance(docs[i], dict) and "content" in docs[i]:
                meta["content"] = docs[i]["content"]
            elif isinstance(docs[i], str):
                meta["content"] = docs[i]
            upsert_data.append((ids[i], vectors[i], meta))
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.index.upsert(vectors=upsert_data))
        self.log_metrics('add_vectors', n)

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
        """Search Pinecone."""
        await self.initialize()
        loop = asyncio.get_event_loop()
        def _search():
            return self.index.query(
                vector=query_vector,
                top_k=k,
                filter=filter_criteria,
                include_metadata=True
            )
        results = await loop.run_in_executor(None, _search)
        search_results = []
        for match in results.matches:
            meta = match.metadata or {}
            doc = meta.get("content", "")
            if metadata_fields:
                meta = {k: v for k, v in meta.items() if k in metadata_fields}
            search_results.append(SearchResult(
                id=match.id,
                score=match.score,
                metadata=meta,
                document=doc
            ))
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from Pinecone."""
        await self.initialize()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.index.delete(ids=ids))
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self) -> None:
        """Clear Pinecone index."""
        await self.initialize()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.index.delete(delete_all=True))
        self.log_metrics('clear', 1)

    async def persist(self, path: str) -> None:
        """Persist Pinecone to disk."""
        self.log_metrics('persist', 1)

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "PineconeBackend":
        """Load Pinecone from disk."""
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