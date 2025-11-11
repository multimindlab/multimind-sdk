"""
FAISS vector store backend implementation.
"""

import logging
import numpy as np
import faiss
from pathlib import Path
import pickle
from typing import List, Dict, Any, Optional

from .base import VectorStoreBackend, VectorStoreConfig, SearchResult

class FAISSBackend(VectorStoreBackend):
    """FAISS vector store backend."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.index = None
        self.metadata = {}
        self.documents = {}
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
                faiss.IndexFlatL2(dimension),
                dimension,
                index_params["nlist"]
            )
        
        if "nprobe" in index_params and hasattr(self.index, "nprobe"):
            self.index.nprobe = index_params["nprobe"]

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to FAISS index."""
        if not self.index:
            await self.initialize()
        
        vectors_array = np.array(vectors).astype("float32")
        self.index.add(vectors_array)
        
        start_id = len(self.metadata)
        for i, (metadata, doc) in enumerate(zip(metadatas, documents)):
            id = ids[i] if ids else f"vec_{start_id + i}"
            self.metadata[id] = metadata
            self.documents[id] = doc

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search FAISS index."""
        if not self.index:
            return []
        
        query_array = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(query_array, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.metadata):  # Check idx >= 0 (FAISS returns -1 for invalid)
                id = f"vec_{idx}"
                # Check if metadata and document exist for this ID
                if id in self.metadata and id in self.documents:
                    results.append(SearchResult(
                        id=id,
                        vector=query_vector,  # FAISS doesn't store vectors
                        metadata=self.metadata[id],
                        document=self.documents[id],
                        score=float(1 / (1 + distance))  # Convert distance to similarity
                    ))
        
        return results

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from FAISS index."""
        if not self.index:
            return
        
        # Create new index
        dimension = self.config.get("dimension", 768)
        new_index = faiss.IndexFlatL2(dimension)
        new_metadata = {}
        new_documents = {}
        
        # Rebuild with remaining vectors
        for i, (id, metadata) in enumerate(self.metadata.items()):
            if id not in ids:
                new_metadata[id] = metadata
                new_documents[id] = self.documents[id]
        
        # Update index
        self.index = new_index
        self.metadata = new_metadata
        self.documents = new_documents

    async def clear(self) -> None:
        """Clear FAISS index."""
        self.index = None
        self.metadata = {}
        self.documents = {}

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
        
        return backend 