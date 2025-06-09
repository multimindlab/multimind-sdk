"""
Pinecone vector store backend implementation.
"""

import logging
from typing import List, Dict, Any, Optional
import pinecone

from ..base import VectorStoreBackend, VectorStoreConfig, SearchResult

class PineconeBackend(VectorStoreBackend):
    """Pinecone vector store backend."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.index = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize Pinecone client and index."""
        pinecone.init(
            **self.config.connection_params
        )
        
        # Get or create index
        index_name = self.config.connection_params.get("index_name", "documents")
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=self.config.dimension,
                metric="cosine"
            )
        
        self.index = pinecone.Index(index_name)

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to Pinecone."""
        if not self.index:
            await self.initialize()
        
        # Prepare vectors
        if not ids:
            ids = [f"doc_{i}" for i in range(len(vectors))]
        
        vectors_to_upsert = []
        for i, (vector, metadata, doc) in enumerate(zip(vectors, metadatas, documents)):
            vectors_to_upsert.append({
                "id": ids[i],
                "values": vector,
                "metadata": {
                    **metadata,
                    "content": doc["content"]
                }
            })
        
        # Upsert vectors
        self.index.upsert(vectors=vectors_to_upsert)

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Pinecone."""
        if not self.index:
            return []
        
        # Execute search
        results = self.index.query(
            vector=query_vector,
            top_k=k,
            filter=filter_criteria,
            include_metadata=True
        )
        
        # Convert to SearchResult format
        search_results = []
        for match in results.matches:
            search_results.append(SearchResult(
                id=match.id,
                vector=query_vector,  # Pinecone doesn't return vectors
                metadata={k: v for k, v in match.metadata.items() if k != "content"},
                document={"content": match.metadata["content"]},
                score=match.score
            ))
        
        return search_results

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from Pinecone."""
        if not self.index:
            return
        
        self.index.delete(ids=ids)

    async def clear(self) -> None:
        """Clear Pinecone index."""
        if not self.index:
            return
        
        self.index.delete(delete_all=True)

    async def persist(self, path: str) -> None:
        """Persist Pinecone to disk."""
        # Pinecone handles persistence internally
        pass

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "PineconeBackend":
        """Load Pinecone from disk."""
        backend = cls(config)
        await backend.initialize()
        return backend 