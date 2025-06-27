from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
import numpy as np

class QdrantBackend(VectorStoreBackend):
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        collection: str = "vectors",
        dim: int = 768,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.host = host or os.environ.get("QDRANT_HOST", "localhost")
        self.port = port or int(os.environ.get("QDRANT_PORT", 6333))
        self.api_key = api_key or os.environ.get("QDRANT_API_KEY")
        self.collection = collection
        self.dim = dim
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self._client = QdrantClient(host=self.host, port=self.port, api_key=self.api_key)
        # Ensure collection exists
        if self.collection not in [c.name for c in self._client.get_collections().collections]:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config={"size": self.dim, "distance": "Cosine"}
            )

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        points = []
        for i in range(n):
            payload = metadatas[i].copy()
            payload["document"] = docs[i]
            points.append(PointStruct(
                id=ids[i],
                vector=np.array(vectors[i], dtype=np.float32),
                payload=payload
            ))
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._client.upsert(collection_name=self.collection, points=points))
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        qdrant_filter = None
        if filter_criteria:
            conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter_criteria.items()]
            qdrant_filter = Filter(must=conditions)
        def _search():
            return self._client.search(
                collection_name=self.collection,
                query_vector=np.array(query_vector, dtype=np.float32),
                limit=k,
                query_filter=qdrant_filter,
                with_payload=True
            )
        results = await loop.run_in_executor(None, _search)
        search_results = [
            SearchResult(
                id=hit.id,
                score=hit.score,
                metadata={k: v for k, v in (hit.payload or {}).items() if k != "document"},
                document=(hit.payload or {}).get("document", "")
            ) for hit in results
        ]
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._client.delete(collection_name=self.collection, points=ids))
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._client.delete(collection_name=self.collection, filter=Filter(must=[])))
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