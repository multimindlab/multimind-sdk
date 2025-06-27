import os
import logging
import asyncio
import numpy as np
import tiledb
from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable

class TileDBVectorStore(VectorStoreBackend):
    def __init__(
        self,
        array_uri: Optional[str] = None,
        dim: int = 768,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.array_uri = array_uri or os.environ.get("TILEDB_ARRAY_URI", "tiledb_vectors")
        self.dim = dim
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self._ensure_array()

    def _ensure_array(self):
        if not tiledb.object_type(self.array_uri):
            dom = tiledb.Domain(
                tiledb.Dim(name="id", domain=(0, 2**63-1), dtype=np.int64, tile=1000)
            )
            schema = tiledb.ArraySchema(
                domain=dom,
                attrs=[
                    tiledb.Attr(name="vector", dtype=np.float32, var=True),
                    tiledb.Attr(name="metadata", dtype="S4096"),
                    tiledb.Attr(name="document", dtype="S4096")
                ]
            )
            tiledb.DenseArray.create(self.array_uri, schema)

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or list(range(n))
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        loop = asyncio.get_event_loop()
        def _add():
            with tiledb.DenseArray(self.array_uri, mode="w") as A:
                A[ids] = {
                    "vector": [np.array(v, dtype=np.float32) for v in vectors],
                    "metadata": [str(m) for m in metadatas],
                    "document": [str(d) for d in docs]
                }
        await loop.run_in_executor(None, _add)
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        def _search():
            with tiledb.DenseArray(self.array_uri, mode="r") as A:
                ids = A.nonempty_domain()[0]
                vectors = A.query(attrs=["vector"]).multi_index[ids[0]:ids[1]+1]["vector"]
                metadatas = A.query(attrs=["metadata"]).multi_index[ids[0]:ids[1]+1]["metadata"]
                documents = A.query(attrs=["document"]).multi_index[ids[0]:ids[1]+1]["document"]
                scores = []
                for i, vec in enumerate(vectors):
                    dist = np.linalg.norm(np.array(vec, dtype=np.float32) - np.array(query_vector, dtype=np.float32))
                    scores.append((i, dist))
                scores.sort(key=lambda x: x[1])
                results = []
                for idx, dist in scores[:k]:
                    results.append(SearchResult(
                        id=ids[0]+idx,
                        score=-dist,
                        metadata=metadatas[idx].decode() if hasattr(metadatas[idx], 'decode') else metadatas[idx],
                        document=documents[idx].decode() if hasattr(documents[idx], 'decode') else documents[idx]
                    ))
                return results
        search_results = await loop.run_in_executor(None, _search)
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        # TileDB does not support deleting individual elements in dense arrays; recommend using sparse arrays for full support
        self.logger.warning("TileDB DenseArray does not support deleting individual vectors. Consider using SparseArray for full support.")
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        def _clear():
            tiledb.remove(self.array_uri)
            self._ensure_array()
        await loop.run_in_executor(None, _clear)
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