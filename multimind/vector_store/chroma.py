"""
Chroma vector store backend implementation.
"""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from .base import VectorStoreBackend, VectorStoreConfig, SearchResult

class ChromaBackend(VectorStoreBackend):
    """Chroma vector store backend."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.client = None
        self.collection = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize Chroma client and collection."""
        settings = Settings(
            **self.config.connection_params.get("settings", {})
        )
        self.client = chromadb.Client(settings)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.connection_params.get("collection_name", "default"),
            metadata={"dimension": self.config.dimension}
        )

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to Chroma collection."""
        if not self.collection:
            await self.initialize()
        
        # Prepare documents and metadatas
        docs = [doc["content"] for doc in documents]
        if not ids:
            ids = [f"doc_{i}" for i in range(len(docs))]
        
        # Add to collection
        self.collection.add(
            embeddings=vectors,
            documents=docs,
            metadatas=metadatas,
            ids=ids
        )

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Chroma collection."""
        if not self.collection:
            return []
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=filter_criteria
        )
        
        # Convert to SearchResult format
        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append(SearchResult(
                id=results["ids"][0][i],
                vector=query_vector,  # Chroma doesn't return vectors
                metadata=results["metadatas"][0][i],
                document={"content": results["documents"][0][i]},
                score=results["distances"][0][i] if "distances" in results else 1.0
            ))
        
        return search_results

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from Chroma collection."""
        if not self.collection:
            return
        
        self.collection.delete(ids=ids)

    async def clear(self) -> None:
        """Clear Chroma collection."""
        if not self.collection:
            return
        
        self.collection.delete(where={})

    async def persist(self, path: str) -> None:
        """Persist Chroma collection to disk."""
        # Chroma persists automatically to the configured directory
        pass

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "ChromaBackend":
        """Load Chroma collection from disk."""
        backend = cls(config)
        await backend.initialize()
        return backend 