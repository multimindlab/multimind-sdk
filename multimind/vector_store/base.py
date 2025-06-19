from typing import Any, Dict, List, Optional, Callable, Union
import abc

class SearchResult:
    def __init__(self, id: str, vector: Any, metadata: Dict[str, Any], document: Any, score: float, explanation: Optional[Dict[str, Any]] = None):
        self.id = id
        self.vector = vector
        self.metadata = metadata
        self.document = document
        self.score = score
        self.explanation = explanation

class VectorStoreConfig:
    def __init__(self, connection_params: Dict[str, Any]):
        self.connection_params = connection_params

class VectorStoreBackend(abc.ABC):
    """
    Abstract base class for all vector store backends.
    Supports hybrid search, custom scoring, metadata filtering, batch ops, live indexing, persistence, monitoring, plugin system, async, error handling, explainability, and extensible config.
    """
    @abc.abstractmethod
    async def add_vectors(self, vectors: List[Any], metadatas: List[Dict[str, Any]], documents: List[Any], ids: Optional[List[str]] = None):
        pass

    @abc.abstractmethod
    async def search(self, query_vector: Any, k: int = 5, query_text: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None, scoring_method: Optional[str] = None, metadata_fields: Optional[List[str]] = None, explain: Optional[bool] = None) -> List[SearchResult]:
        pass

    @abc.abstractmethod
    async def delete_vectors(self, ids: List[str]):
        pass

    @abc.abstractmethod
    async def clear(self):
        pass

    @abc.abstractmethod
    async def persist(self, path: str):
        pass

    @classmethod
    @abc.abstractmethod
    async def load(cls, path: str, config: VectorStoreConfig):
        pass

    @abc.abstractmethod
    def register_plugin(self, name: str, plugin: Callable):
        pass

    @abc.abstractmethod
    async def _run_plugin(self, name: str, *args, **kwargs):
        pass

    @abc.abstractmethod
    def log_metrics(self, metric_name: str, value: Any):
        pass

    @abc.abstractmethod
    async def _with_retries(self, func, *args, **kwargs):
        pass
