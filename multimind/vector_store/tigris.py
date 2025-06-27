import os
import logging
import asyncio
import numpy as np
from tigrisdb import TigrisClient
from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable

class TigrisVectorStore(VectorStoreBackend):
    def __init__(
        self,
        project: Optional[str] = None,
        database: str = "test_db",
        collection: str = "vectors",
        dim: int = 768,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.project = project or os.environ.get("TIGRIS_PROJECT")
        self.database = database
        self.collection = collection
        self.dim = dim
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self.client = TigrisClient(project=self.project)
        self._ensure_collection()

    def _ensure_collection(self):
        # Create database and collection if not exists
        try:
            db = self.client.get_database(self.database)
            if self.collection not in db.list_collections():
                db.create_collection(self.collection, schema={
                    "id": "string",
                    "vector": ["float" for _ in range(self.dim)],
                    "metadata": "object",
                    "document": "string"
                })
        except Exception as e:
            self.logger.warning(f"Collection ensure failed or already exists: {e}")

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        loop = asyncio.get_event_loop()
        def _add():
            db = self.client.get_database(self.database)
            col = db.get_collection(self.collection)
            for i in range(n):
                doc = {
                    "id": ids[i],
                    "vector": list(map(float, vectors[i])),
                    "metadata": metadatas[i],
                    "document": docs[i]
                }
                col.insert_one(doc)
        await loop.run_in_executor(None, _add)
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        def _search():
            db = self.client.get_database(self.database)
            col = db.get_collection(self.collection)
            # Tigris does not natively support vector search, so we do it in Python for now
            all_docs = list(col.find({}))
            scores = []
            for doc in all_docs:
                vec = np.array(doc["vector"], dtype=np.float32)
                qvec = np.array(query_vector, dtype=np.float32)
                dist = np.linalg.norm(vec - qvec)
                scores.append((doc, dist))
            scores.sort(key=lambda x: x[1])
            results = []
            for doc, dist in scores[:k]:
                results.append(SearchResult(
                    id=doc["id"],
                    score=-dist,
                    metadata=doc.get("metadata"),
                    document=doc.get("document")
                ))
            return results
        search_results = await loop.run_in_executor(None, _search)
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        def _delete():
            db = self.client.get_database(self.database)
            col = db.get_collection(self.collection)
            for id_ in ids:
                col.delete_one({"id": id_})
        await loop.run_in_executor(None, _delete)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        def _clear():
            db = self.client.get_database(self.database)
            col = db.get_collection(self.collection)
            col.delete_many({})
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