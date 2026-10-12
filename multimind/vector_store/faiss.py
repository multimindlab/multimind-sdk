"""
FAISS vector store backend implementation.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from .base import SearchResult, VectorStoreBackend, VectorStoreConfig


class FAISSBackend(VectorStoreBackend):
    """FAISS vector store backend."""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.index = None
        self.metadata = {}
        self.documents = {}
        # Insertion-ordered id list (FAISS positions) and raw vectors, kept so
        # search can map positions to caller ids and delete can rebuild.
        self._id_order: List[str] = []
        self._vectors: Dict[str, np.ndarray] = {}
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize FAISS index."""
        # Get dimension from config
        dimension = self.config.get("dimension", 768)

        # Get index_params from config, default to empty dict
        index_params = self.config.get("index_params", {})

        # Determine index type from config
        index_type = self.config.get("index_type", "flat")

        # Create index based on type
        if index_type == "flat":
            metric = self.config.get("metric", "l2")
            if metric == "cosine":
                # For cosine similarity, we need to normalize vectors
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
            else:
                self.index = faiss.IndexFlatL2(dimension)
        else:
            # Default to L2 flat index
            self.index = faiss.IndexFlatL2(dimension)

        # Apply advanced index types if specified
        if "nlist" in index_params:
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatL2(dimension), dimension, index_params["nlist"]
            )

        if "nprobe" in index_params and hasattr(self.index, "nprobe"):
            self.index.nprobe = index_params["nprobe"]

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add vectors to FAISS index."""
        if not self.index:
            await self.initialize()

        vectors_array = np.array(vectors).astype("float32")
        self.index.add(vectors_array)

        start_id = len(self._id_order)
        for i, (metadata, doc) in enumerate(zip(metadatas, documents)):
            id = ids[i] if ids else f"vec_{start_id + i}"
            self.metadata[id] = metadata
            self.documents[id] = doc
            self._id_order.append(id)
            self._vectors[id] = vectors_array[i]

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search FAISS index."""
        if not self.index:
            return []

        def _matches(metadata: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
            for key, expected in criteria.items():
                if key == "id":
                    continue
                if metadata.get(key) != expected:
                    return False
            return True

        # Direct id lookup. FAISS search order is not an id index, so a filter
        # on id must not depend on the query vector ranking.
        if filter_criteria and "id" in filter_criteria:
            wanted = str(filter_criteria["id"])
            if wanted not in self.metadata or wanted not in self.documents:
                return []
            if not _matches(self.metadata[wanted], filter_criteria):
                return []
            stored = self._vectors.get(wanted)
            vector = stored.tolist() if stored is not None else query_vector
            return [
                SearchResult(
                    id=wanted,
                    vector=vector,
                    metadata=self.metadata[wanted],
                    document=self.documents[wanted],
                    score=1.0,
                )
            ]

        ntotal = len(self._id_order)
        search_k = ntotal if filter_criteria else k
        search_k = max(search_k, 1)
        query_array = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(query_array, search_k)

        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self._id_order):
                id = self._id_order[idx]
                if id in self.metadata and id in self.documents:
                    if filter_criteria and not _matches(self.metadata[id], filter_criteria):
                        continue
                    results.append(
                        SearchResult(
                            id=id,
                            vector=query_vector,
                            metadata=self.metadata[id],
                            document=self.documents[id],
                            score=float(1 / (1 + distance)),
                        )
                    )
                    if len(results) >= k:
                        break

        return results

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from FAISS index (rebuilds the index without them)."""
        if not self.index:
            return

        to_delete = set(ids)
        remaining = [id for id in self._id_order if id not in to_delete]

        # Rebuild the index from the retained raw vectors.
        await self.initialize()
        if remaining:
            self.index.add(np.stack([self._vectors[id] for id in remaining]))

        self.metadata = {id: self.metadata[id] for id in remaining}
        self.documents = {id: self.documents[id] for id in remaining}
        self._vectors = {id: self._vectors[id] for id in remaining}
        self._id_order = remaining

    async def clear(self) -> None:
        """Clear FAISS index."""
        self.index = None
        self.metadata = {}
        self.documents = {}
        self._id_order = []
        self._vectors = {}

    async def persist(self, path: str) -> None:
        """Persist FAISS index to disk."""
        if not self.index:
            return

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save index
        faiss.write_index(self.index, str(path / "index.faiss"))

        # Save metadata and documents
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        with open(path / "vectors.pkl", "wb") as f:
            pickle.dump({"id_order": self._id_order, "vectors": self._vectors}, f)

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "FAISSBackend":
        """Load FAISS index from disk."""
        path = Path(path)

        backend = cls(config)

        # Load index
        if (path / "index.faiss").exists():
            backend.index = faiss.read_index(str(path / "index.faiss"))

        # Load metadata and documents
        if (path / "metadata.pkl").exists():
            with open(path / "metadata.pkl", "rb") as f:
                backend.metadata = pickle.load(f)
        if (path / "documents.pkl").exists():
            with open(path / "documents.pkl", "rb") as f:
                backend.documents = pickle.load(f)
        if (path / "vectors.pkl").exists():
            with open(path / "vectors.pkl", "rb") as f:
                state = pickle.load(f)
            backend._id_order = state["id_order"]
            backend._vectors = state["vectors"]
        else:
            # Older persisted stores: fall back to metadata insertion order.
            backend._id_order = list(backend.metadata.keys())

        return backend
