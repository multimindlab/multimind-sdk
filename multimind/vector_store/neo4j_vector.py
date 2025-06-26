from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
from neo4j import GraphDatabase, basic_auth
import numpy as np

class Neo4jVectorBackend(VectorStoreBackend):
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
        label: str = "VectorNode",
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self.database = database
        self.label = label
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        self._driver = GraphDatabase.driver(self.uri, auth=basic_auth(self.user, self.password))

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        n = len(vectors)
        ids = ids or [str(i) for i in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]
        docs = documents or ["" for _ in range(n)]
        loop = asyncio.get_event_loop()
        def _add():
            with self._driver.session(database=self.database) as session:
                for i in range(n):
                    session.run(
                        f"""
                        MERGE (n:{self.label} {{id: $id}})
                        SET n.vector = $vector, n.metadata = $metadata, n.document = $document
                        """,
                        id=ids[i],
                        vector=list(map(float, vectors[i])),
                        metadata=metadatas[i],
                        document=docs[i]
                    )
        await loop.run_in_executor(None, _add)
        self.log_metrics('add_vectors', n)

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        # Neo4j does not natively support vector search, but you can compute cosine similarity in Cypher
        loop = asyncio.get_event_loop()
        filter_cypher = ""
        if filter_criteria:
            filter_clauses = [f"n.metadata.{k} = '{v}'" for k, v in filter_criteria.items()]
            filter_cypher = " AND ".join(filter_clauses)
        where_clause = f"WHERE {filter_cypher}" if filter_cypher else ""
        query_vec = np.array(query_vector, dtype=np.float32)
        def _search():
            with self._driver.session(database=self.database) as session:
                cypher = f"""
                MATCH (n:{self.label})
                {where_clause}
                WITH n, n.vector AS v
                WITH n, reduce(s = 0.0, i IN range(0, size(v)-1) | s + v[i]*{query_vec.tolist()}[i]) AS dot,
                     reduce(s = 0.0, i IN range(0, size(v)-1) | s + v[i]*v[i]) AS norm_n,
                     reduce(s = 0.0, i IN range(0, size({query_vec.tolist()})-1) | s + {query_vec.tolist()}[i]*{query_vec.tolist()}[i]) AS norm_q
                WITH n, dot/(sqrt(norm_n)*sqrt(norm_q)) AS score
                RETURN n.id AS id, n.metadata AS metadata, n.document AS document, score
                ORDER BY score DESC
                LIMIT {k}
                """
                result = session.run(cypher)
                return [
                    SearchResult(
                        id=record["id"],
                        score=record["score"],
                        metadata=record["metadata"],
                        document=record["document"]
                    ) for record in result
                ]
        search_results = await loop.run_in_executor(None, _search)
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        def _delete():
            with self._driver.session(database=self.database) as session:
                session.run(f"MATCH (n:{self.label}) WHERE n.id IN $ids DETACH DELETE n", ids=ids)
        await loop.run_in_executor(None, _delete)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        loop = asyncio.get_event_loop()
        def _clear():
            with self._driver.session(database=self.database) as session:
                session.run(f"MATCH (n:{self.label}) DETACH DELETE n")
        await loop.run_in_executor(None, _clear)
        self.log_metrics('clear', 1)

    async def persist(self, path):
        # Neo4j is persistent by default
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