from typing import Any, Dict, Optional, List

class UnifiedMemoryStore:
    """
    Abstracts access to multiple memory backends (vector, graph, key-value, etc.).
    Allows switching, routing, and unified querying for developer-friendly workflows.
    """
    def __init__(self, vector_store=None, graph_store=None, kv_store=None):
        self.backends = {
            'vector': vector_store,
            'graph': graph_store,
            'kv': kv_store
        }
        self.active_backend = 'vector' if vector_store else (
            'graph' if graph_store else (
                'kv' if kv_store else None))

    def switch_backend(self, backend: str):
        """
        Switch the active memory backend (vector, graph, kv).
        """
        if backend not in self.backends or self.backends[backend] is None:
            raise ValueError(f"Backend '{backend}' is not available.")
        self.active_backend = backend

    def add(self, *args, **kwargs) -> Any:
        """
        Add an entry to the active backend.
        """
        backend = self.backends.get(self.active_backend)
        if backend is None:
            raise RuntimeError("No active memory backend.")
        if hasattr(backend, 'add'):
            return backend.add(*args, **kwargs)
        elif hasattr(backend, 'add_triple'):
            return backend.add_triple(*args, **kwargs)
        else:
            raise NotImplementedError(f"Add not implemented for backend {self.active_backend}.")

    def get(self, *args, **kwargs) -> Any:
        """
        Get an entry from the active backend.
        """
        backend = self.backends.get(self.active_backend)
        if backend is None:
            raise RuntimeError("No active memory backend.")
        if hasattr(backend, 'get'):
            return backend.get(*args, **kwargs)
        elif hasattr(backend, 'get_triples'):
            return backend.get_triples(*args, **kwargs)
        else:
            raise NotImplementedError(f"Get not implemented for backend {self.active_backend}.")

    def query(self, *args, **kwargs) -> Any:
        """
        Query the active backend (supports vector search, graph query, or key lookup).
        """
        backend = self.backends.get(self.active_backend)
        if backend is None:
            raise RuntimeError("No active memory backend.")
        if hasattr(backend, 'query'):
            return backend.query(*args, **kwargs)
        elif hasattr(backend, 'find_triples'):
            return backend.find_triples(*args, **kwargs)
        else:
            raise NotImplementedError(f"Query not implemented for backend {self.active_backend}.")

    def available_backends(self) -> List[str]:
        """
        List all available memory backends.
        """
        return [k for k, v in self.backends.items() if v is not None] 