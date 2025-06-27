import os
import logging
import asyncio
import numpy as np
import grpc
from vald.v1.vald import insert_pb2_grpc, search_pb2_grpc, update_pb2_grpc, remove_pb2_grpc, flush_pb2_grpc
from vald.v1.payload import payload_pb2
from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable

class ValdVectorStore(VectorStoreBackend):
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        dim: int = 768,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.host = host or os.environ.get("VALD_HOST", "localhost")
        self.port = port or int(os.environ.get("VALD_PORT", 8081))
        self.dim = dim
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self.channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        self.istub = insert_pb2_grpc.InsertStub(self.channel)
        self.sstub = search_pb2_grpc.SearchStub(self.channel)
        self.ustub = update_pb2_grpc.UpdateStub(self.channel)
        self.rstub = remove_pb2_grpc.RemoveStub(self.channel)
        self.fstub = flush_pb2_grpc.FlushStub(self.channel)

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        loop = asyncio.get_event_loop()
        def _add():
            for i in range(n):
                vec = payload_pb2.Object.Vector(id=ids[i], vector=list(map(float, vectors[i])))
                icfg = payload_pb2.Insert.Config(skip_strict_exist_check=True)
                self.istub.Insert(payload_pb2.Insert.Request(vector=vec, config=icfg))
        await loop.run_in_executor(None, _add)
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        def _search():
            scfg = payload_pb2.Search.Config(num=k, radius=-1.0, epsilon=0.01, timeout=3000000000)
            res = self.sstub.Search(payload_pb2.Search.Request(vector=list(map(float, query_vector)), config=scfg))
            results = []
            for hit in res:
                results.append(SearchResult(
                    id=hit.id,
                    score=hit.distance,
                    metadata=None,
                    document=None
                ))
            return results
        search_results = await loop.run_in_executor(None, _search)
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        def _delete():
            for id_ in ids:
                rid = payload_pb2.Object.ID(id=id_)
                rcfg = payload_pb2.Remove.Config(skip_strict_exist_check=True)
                self.rstub.Remove(payload_pb2.Remove.Request(id=rid, config=rcfg))
        await loop.run_in_executor(None, _delete)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        def _clear():
            self.fstub.Flush(payload_pb2.Flush.Request())
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