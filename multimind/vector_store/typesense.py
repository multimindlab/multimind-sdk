import os
import logging
import asyncio
import numpy as np
import typesense
from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable

class TypesenseVectorStore(VectorStoreBackend):
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
        self.host = host or os.environ.get("TYPESENSE_HOST", "localhost")
        self.port = port or int(os.environ.get("TYPESENSE_PORT", 8108))
        self.api_key = api_key or os.environ.get("TYPESENSE_API_KEY")
        self.collection = collection
        self.dim = dim
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self.client = typesense.Client({
            'nodes': [{
                'host': self.host,
                'port': self.port,
                'protocol': 'http'
            }],
            'api_key': self.api_key,
            'connection_timeout_seconds': 2
        })
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            if self.collection not in [c['name'] for c in self.client.collections.retrieve()]:
                self.client.collections.create({
                    'name': self.collection,
                    'fields': [
                        {'name': 'id', 'type': 'string'},
                        {'name': 'vector', 'type': 'float[]', 'num_dim': self.dim},
                        {'name': 'metadata', 'type': 'object', 'optional': True},
                        {'name': 'document', 'type': 'string', 'optional': True}
                    ],
                    'default_sorting_field': 'id'
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
            docs_to_add = []
            for i in range(n):
                docs_to_add.append({
                    'id': ids[i],
                    'vector': list(map(float, vectors[i])),
                    'metadata': metadatas[i],
                    'document': docs[i]
                })
            self.client.collections[self.collection].documents.import_(docs_to_add, {'action': 'upsert'})
        await loop.run_in_executor(None, _add)
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        def _search():
            try:
                search_params = {
                    'q': '*',
                    'vector_query': f'vector:([{','.join(map(str, query_vector))}], k:{k})',
                    'query_by': 'document',
                    'per_page': k
                }
                if filter_criteria:
                    filters = [f"metadata.{key}:={repr(value)}" for key, value in filter_criteria.items()]
                    search_params['filter_by'] = ' && '.join(filters)
                res = self.client.collections[self.collection].documents.search(search_params)
                hits = res.get('hits', [])
            except Exception as e:
                self.logger.error(f"Search failed: {e}")
                hits = []
            search_results = []
            for hit in hits:
                doc = hit['document']
                search_results.append(SearchResult(
                    id=doc.get('id'),
                    score=hit.get('text_match', 0),
                    metadata=doc.get('metadata'),
                    document=doc.get('document')
                ))
            return search_results
        search_results = await loop.run_in_executor(None, _search)
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        def _delete():
            for id_ in ids:
                self.client.collections[self.collection].documents[id_].delete()
        await loop.run_in_executor(None, _delete)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        def _clear():
            self.client.collections[self.collection].documents.delete({'filter_by': '*'})
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