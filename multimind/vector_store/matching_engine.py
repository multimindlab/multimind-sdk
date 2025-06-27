from .base import VectorStoreBackend, VectorStoreConfig, SearchResult
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import asyncio
from google.cloud import aiplatform

class MatchingEngineBackend(VectorStoreBackend):
    def __init__(
        self,
        project: Optional[str] = None,
        location: Optional[str] = None,
        index_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        metrics_enabled: bool = False,
        plugin_registry: Optional[Dict[str, Callable]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.index_id = index_id or os.environ.get("MATCHING_ENGINE_INDEX_ID")
        self.endpoint_id = endpoint_id or os.environ.get("MATCHING_ENGINE_ENDPOINT_ID")
        self.metrics_enabled = metrics_enabled
        self.plugin_registry = plugin_registry or {}
        self.retry_policy = retry_policy or {"retries": 3}
        self.logger = logging.getLogger(__name__)
        if not self.project or not self.index_id or not self.endpoint_id:
            raise ValueError("project, index_id, and endpoint_id must be provided for Matching Engine.")
        aiplatform.init(project=self.project, location=self.location)
        self.index = aiplatform.MatchingEngineIndex(index_name=self.index_id)
        self.endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=self.endpoint_id)

    async def add_vectors(self, vectors, metadatas, documents, ids=None):
        # Matching Engine expects embeddings and optional metadata
        # This is a sync API, so run in thread
        loop = asyncio.get_event_loop()
        # Prepare datapoints
        datapoints = []
        for i, vector in enumerate(vectors):
            dp = {
                "datapoint_id": ids[i] if ids else str(i),
                "feature_vector": vector,
                "restricts": [],
                "crowding_tag": "",
            }
            if metadatas and i < len(metadatas):
                dp["restricts"] = metadatas[i]
            datapoints.append(dp)
        await loop.run_in_executor(None, self.index.upsert_datapoints, datapoints)
        self.log_metrics('add_vectors', len(datapoints))

    async def search(self, query_vector, k=5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        # Only vector search is supported
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self.endpoint.match,
            [query_vector],
            k,
            None,  # deployed_index_id
            None,  # traffic_split
            None,  # filter
        )
        search_results = []
        for match in results[0].neighbors:
            search_results.append(
                SearchResult(
                    id=match.datapoint.datapoint_id,
                    score=match.distance,
                    metadata={},
                    document=""
                )
            )
        self.log_metrics('search', len(search_results))
        return search_results

    async def delete_vectors(self, ids):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.index.remove_datapoints, ids)
        self.log_metrics('delete_vectors', len(ids))

    async def clear(self):
        # No direct clear; must remove all datapoints by listing and deleting
        loop = asyncio.get_event_loop()
        datapoints = await loop.run_in_executor(None, self.index.list_datapoints)
        all_ids = [dp.datapoint_id for dp in datapoints]
        if all_ids:
            await loop.run_in_executor(None, self.index.remove_datapoints, all_ids)
        self.log_metrics('clear', len(all_ids))

    async def persist(self, path):
        # Matching Engine is managed; this is a no-op
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