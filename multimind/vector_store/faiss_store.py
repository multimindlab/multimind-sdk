"""
FAISS vector store implementation.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import faiss
from .base import VectorStore, VectorStoreConfig

class FAISSVectorStore(VectorStore):
    """FAISS vector store implementation."""
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize the FAISS store."""
        self.config = config
        self.dimension = config.dimension
        
        # Create FAISS index
        if config.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif config.index_type == "ivf":
            nlist = config.metadata.get("nlist", 100)
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatL2(self.dimension),
                self.dimension,
                nlist
            )
        elif config.index_type == "hnsw":
            M = config.metadata.get("M", 16)
            self.index = faiss.IndexHNSWFlat(
                self.dimension,
                M,
                faiss.METRIC_L2
            )
        else:
            raise ValueError(f"Unsupported index type: {config.index_type}")
        
        # Store metadata
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
    
    async def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        **kwargs
    ) -> List[str]:
        """Add vectors to the store."""
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        # Convert vectors to float32
        vectors = [v.astype(np.float32) for v in vectors]
        
        # Add vectors to index
        vector_ids = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            vector_id = str(len(self.metadata_store))
            self.index.add(vector.reshape(1, -1))
            self.metadata_store[vector_id] = meta
            vector_ids.append(vector_id)
        
        return vector_ids
    
    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        # Convert query to float32
        query_vector = query_vector.astype(np.float32)
        
        # Search index
        distances, indices = self.index.search(
            query_vector.reshape(1, -1),
            k
        )
        
        # Get results with metadata
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx != -1:  # FAISS returns -1 for empty slots
                vector_id = str(idx)
                result = {
                    "vector_id": vector_id,
                    "distance": float(distance),
                    "metadata": self.metadata_store.get(vector_id, {})
                }
                results.append(result)
        
        return results
    
    async def delete_vectors(
        self,
        vector_ids: List[str],
        **kwargs
    ) -> bool:
        """Delete vectors from the store."""
        # FAISS doesn't support direct deletion
        # We'll mark vectors as deleted in metadata
        for vector_id in vector_ids:
            if vector_id in self.metadata_store:
                self.metadata_store[vector_id]["deleted"] = True
        return True
    
    async def get_vector(
        self,
        vector_id: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Get a vector by ID."""
        if vector_id not in self.metadata_store:
            return None
        
        # FAISS doesn't support direct vector retrieval
        # We can only return metadata
        return {
            "vector_id": vector_id,
            "metadata": self.metadata_store[vector_id]
        }
    
    async def update_metadata(
        self,
        vector_id: str,
        metadata: Dict[str, Any],
        **kwargs
    ) -> bool:
        """Update metadata for a vector."""
        if vector_id not in self.metadata_store:
            return False
        
        self.metadata_store[vector_id].update(metadata)
        return True
    
    def save(self, filepath: str):
        """Save the index to disk."""
        faiss.write_index(self.index, filepath)
    
    def load(self, filepath: str):
        """Load the index from disk."""
        self.index = faiss.read_index(filepath) 