from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
from pymongo import MongoClient
from pymongo.errors import PyMongoError

class MongoDBAtlasBackend(VectorStoreBackend):
    def __init__(
        self,
        uri: Optional[str] = None,
        database: str = "vector_db",
        collection: str = "vectors",
        vector_field: str = "vector",
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.uri = uri or os.environ.get("MONGODB_ATLAS_URI")
        self.database = database
        self.collection_name = collection
        self.vector_field = vector_field
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self._client = MongoClient(self.uri)
        self._db = self._client[self.database]
        self._collection = self._db[self.collection_name]
        # Ensure vector index exists (Atlas vector search index)
        # This is a no-op if already exists
        try:
            self._collection.create_index([(self.vector_field, "text")])
        except Exception:
            pass

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        docs = []
        for i, vector in enumerate(vectors):
            doc = {
                "_id": ids[i] if ids else str(i),
                self.vector_field: vector,
                "metadata": metadatas[i] if metadatas and i < len(metadatas) else {},
                "document": documents[i] if documents and i < len(documents) else ""
            }
            docs.append(doc)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._collection.insert_many(docs, ordered=False))
        self.log_metrics('add_vectors', len(docs))

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        # MongoDB Atlas vector search (requires Atlas Search index)
        loop = asyncio.get_event_loop()
        pipeline = [
            {"$search": {
                "index": "default",
                "knnBeta": {
                    "vector": query_vector,
                    "path": self.vector_field,
                    "k": k
                }
            }}
        ]
        if filter_criteria:
            pipeline.append({"$match": filter_criteria})
        if metadata_fields:
            projection = {field: 1 for field in metadata_fields}
            projection.update({"_id": 1, "score": {"$meta": "searchScore"}, "document": 1, "metadata": 1})
            pipeline.append({"$project": projection})
        else:
            pipeline.append({"$project": {"_id": 1, "score": {"$meta": "searchScore"}, "document": 1, "metadata": 1}})
        results = await loop.run_in_executor(None, lambda: list(self._collection.aggregate(pipeline)))
        search_results = [
            SearchResult(
                id=doc.get("_id"),
                score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {}),
                document=doc.get("document", "")
            ) for doc in results
        ]
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._collection.delete_many({"_id": {"$in": ids}}))
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._collection.delete_many, {})
        self.log_metrics('clear', 1)

    async def persist(self, path):
        # MongoDB Atlas is persistent by default
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