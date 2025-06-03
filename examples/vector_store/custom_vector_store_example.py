"""
Example demonstrating how to customize the vector store for different use cases.
"""

import asyncio
import os
import numpy as np
from typing import Dict, List, Any
from multimind.core.provider import ProviderConfig
from multimind.core.router import Router, TaskType, TaskConfig, RoutingStrategy
from multimind.vector_store.base import VectorStore, VectorStoreConfig, VectorStoreFactory
from multimind.providers.openai import OpenAIProvider

class CustomMetadataVectorStore(VectorStore):
    """Vector store with custom metadata handling."""
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize with custom metadata fields."""
        self.config = config
        self.dimension = config.dimension
        self.metadata_fields = config.metadata.get("required_fields", [])
        self.store: Dict[str, Dict[str, Any]] = {}
    
    async def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        **kwargs
    ) -> List[str]:
        """Add vectors with metadata validation."""
        # Validate required metadata fields
        for meta in metadata:
            for field in self.metadata_fields:
                if field not in meta:
                    raise ValueError(f"Missing required metadata field: {field}")
        
        # Add vectors with metadata
        vector_ids = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            vector_id = f"vec_{len(self.store)}"
            self.store[vector_id] = {
                "vector": vector,
                "metadata": meta
            }
            vector_ids.append(vector_id)
        
        return vector_ids
    
    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        metadata_filter: Dict[str, Any] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search with metadata filtering."""
        results = []
        
        # Calculate similarities
        for vector_id, data in self.store.items():
            vector = data["vector"]
            metadata = data["metadata"]
            
            # Apply metadata filter if provided
            if metadata_filter:
                if not all(
                    metadata.get(k) == v 
                    for k, v in metadata_filter.items()
                ):
                    continue
            
            # Calculate cosine similarity
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            
            results.append({
                "vector_id": vector_id,
                "distance": float(1 - similarity),  # Convert to distance
                "metadata": metadata
            })
        
        # Sort by distance and return top k
        results.sort(key=lambda x: x["distance"])
        return results[:k]
    
    async def delete_vectors(
        self,
        vector_ids: List[str],
        **kwargs
    ) -> bool:
        """Delete vectors."""
        for vector_id in vector_ids:
            self.store.pop(vector_id, None)
        return True
    
    async def get_vector(
        self,
        vector_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Get vector by ID."""
        return self.store.get(vector_id)
    
    async def update_metadata(
        self,
        vector_id: str,
        metadata: Dict[str, Any],
        **kwargs
    ) -> bool:
        """Update metadata with validation."""
        if vector_id not in self.store:
            return False
        
        # Validate required fields
        for field in self.metadata_fields:
            if field not in metadata:
                raise ValueError(f"Missing required metadata field: {field}")
        
        self.store[vector_id]["metadata"].update(metadata)
        return True

class TimeSeriesVectorStore(VectorStore):
    """Vector store with time-series support."""
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize with time-series support."""
        self.config = config
        self.dimension = config.dimension
        self.store: Dict[str, Dict[str, Any]] = {}
        self.time_index: Dict[str, List[str]] = {}  # timestamp -> vector_ids
    
    async def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        **kwargs
    ) -> List[str]:
        """Add vectors with timestamp indexing."""
        vector_ids = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            vector_id = f"vec_{len(self.store)}"
            timestamp = meta.get("timestamp")
            
            if timestamp:
                if timestamp not in self.time_index:
                    self.time_index[timestamp] = []
                self.time_index[timestamp].append(vector_id)
            
            self.store[vector_id] = {
                "vector": vector,
                "metadata": meta
            }
            vector_ids.append(vector_id)
        
        return vector_ids
    
    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        time_range: tuple = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search with time range filtering."""
        results = []
        
        # Get vector IDs in time range
        vector_ids = set()
        if time_range:
            start_time, end_time = time_range
            for timestamp, ids in self.time_index.items():
                if start_time <= timestamp <= end_time:
                    vector_ids.update(ids)
        else:
            vector_ids = set(self.store.keys())
        
        # Calculate similarities for filtered vectors
        for vector_id in vector_ids:
            data = self.store[vector_id]
            vector = data["vector"]
            
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            
            results.append({
                "vector_id": vector_id,
                "distance": float(1 - similarity),
                "metadata": data["metadata"]
            })
        
        # Sort by distance and return top k
        results.sort(key=lambda x: x["distance"])
        return results[:k]
    
    async def delete_vectors(
        self,
        vector_ids: List[str],
        **kwargs
    ) -> bool:
        """Delete vectors and update time index."""
        for vector_id in vector_ids:
            if vector_id in self.store:
                metadata = self.store[vector_id]["metadata"]
                timestamp = metadata.get("timestamp")
                
                if timestamp and timestamp in self.time_index:
                    self.time_index[timestamp].remove(vector_id)
                    if not self.time_index[timestamp]:
                        del self.time_index[timestamp]
                
                del self.store[vector_id]
        
        return True
    
    async def get_vector(
        self,
        vector_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Get vector by ID."""
        return self.store.get(vector_id)
    
    async def update_metadata(
        self,
        vector_id: str,
        metadata: Dict[str, Any],
        **kwargs
    ) -> bool:
        """Update metadata and time index."""
        if vector_id not in self.store:
            return False
        
        old_timestamp = self.store[vector_id]["metadata"].get("timestamp")
        new_timestamp = metadata.get("timestamp")
        
        # Update time index if timestamp changed
        if old_timestamp != new_timestamp:
            if old_timestamp and old_timestamp in self.time_index:
                self.time_index[old_timestamp].remove(vector_id)
                if not self.time_index[old_timestamp]:
                    del self.time_index[old_timestamp]
            
            if new_timestamp:
                if new_timestamp not in self.time_index:
                    self.time_index[new_timestamp] = []
                self.time_index[new_timestamp].append(vector_id)
        
        self.store[vector_id]["metadata"].update(metadata)
        return True

async def main():
    # Initialize provider
    openai_config = ProviderConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1"
    )
    openai_provider = OpenAIProvider(openai_config)
    
    # Initialize router
    router = Router()
    router.register_provider("openai", openai_provider)
    
    # Example 1: Custom Metadata Vector Store
    print("\nExample 1: Custom Metadata Vector Store")
    
    metadata_config = VectorStoreConfig(
        dimension=1536,
        metadata={
            "required_fields": ["category", "language", "source"]
        }
    )
    
    metadata_store = CustomMetadataVectorStore(metadata_config)
    
    # Add vectors with required metadata
    vectors = [np.random.rand(1536) for _ in range(3)]
    metadata = [
        {
            "category": "news",
            "language": "en",
            "source": "reuters",
            "text": "First article about technology"
        },
        {
            "category": "blog",
            "language": "en",
            "source": "medium",
            "text": "Second article about AI"
        },
        {
            "category": "news",
            "language": "es",
            "source": "elpais",
            "text": "Third article about science"
        }
    ]
    
    vector_ids = await metadata_store.add_vectors(vectors, metadata)
    
    # Search with metadata filter
    query_vector = np.random.rand(1536)
    results = await metadata_store.search(
        query_vector,
        metadata_filter={"category": "news", "language": "en"}
    )
    
    print("\nSearch results with metadata filter:")
    for result in results:
        print(f"- {result['metadata']['text']}")
    
    # Example 2: Time Series Vector Store
    print("\nExample 2: Time Series Vector Store")
    
    time_config = VectorStoreConfig(
        dimension=1536
    )
    
    time_store = TimeSeriesVectorStore(time_config)
    
    # Add vectors with timestamps
    vectors = [np.random.rand(1536) for _ in range(3)]
    metadata = [
        {
            "timestamp": "2024-01-01",
            "text": "First article from January"
        },
        {
            "timestamp": "2024-02-01",
            "text": "Second article from February"
        },
        {
            "timestamp": "2024-03-01",
            "text": "Third article from March"
        }
    ]
    
    vector_ids = await time_store.add_vectors(vectors, metadata)
    
    # Search with time range
    query_vector = np.random.rand(1536)
    results = await time_store.search(
        query_vector,
        time_range=("2024-01-01", "2024-02-15")
    )
    
    print("\nSearch results within time range:")
    for result in results:
        print(f"- {result['metadata']['text']}")

if __name__ == "__main__":
    asyncio.run(main()) 