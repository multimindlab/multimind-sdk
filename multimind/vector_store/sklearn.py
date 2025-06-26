from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import logging
import asyncio
import numpy as np
from sklearn.neighbors import NearestNeighbors

class SklearnBackend(VectorStoreBackend):
    def __init__(
        self,
        metric: str = "cosine",
        dim: int = 768,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.metric = metric
        self.dim = dim
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self._vectors = []
        self._ids = []
        self._metadatas = []
        self._documents = []
        self._nn = None

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(len(self._vectors), len(self._vectors) + n)]
        metadatas = metadatas or [{} for _ in range(n)]
        documents = documents or ["" for _ in range(n)]
        self._vectors.extend([np.array(v, dtype=np.float32) for v in vectors])
        self._ids.extend(ids)
        self._metadatas.extend(metadatas)
        self._documents.extend(documents)
        self._fit_nn()
        self.log_metrics('add_vectors', n)

    def _fit_nn(self):
        if self._vectors:
            self._nn = NearestNeighbors(metric=self.metric)
            self._nn.fit(np.stack(self._vectors))
        else:
            self._nn = None

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        if not self._nn:
            return []
        query_vec = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        loop = asyncio.get_event_loop()
        dists, indices = await loop.run_in_executor(None, lambda: self._nn.kneighbors(query_vec, n_neighbors=k))
        results = []
        for rank, idx in enumerate(indices[0]):
            meta = self._metadatas[idx]
            doc = self._documents[idx]
            if filter_criteria:
                if not all(meta.get(k) == v for k, v in filter_criteria.items()):
                    continue
            if metadata_fields:
                meta = {k: v for k, v in meta.items() if k in metadata_fields}
            results.append(SearchResult(
                id=self._ids[idx],
                score=-dists[0][rank],  # negative distance for similarity
                metadata=meta,
                document=doc
            ))
        self.log_metrics('search', len(results))
        return results[:k]

    async def delete_vectors(self, ids):
        id_set = set(ids)
        keep = [i for i, id_ in enumerate(self._ids) if id_ not in id_set]
        self._vectors = [self._vectors[i] for i in keep]
        self._ids = [self._ids[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]
        self._documents = [self._documents[i] for i in keep]
        self._fit_nn()
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        self._vectors = []
        self._ids = []
        self._metadatas = []
        self._documents = []
        self._fit_nn()
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