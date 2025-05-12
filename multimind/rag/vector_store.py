"""
Vector store implementations using FAISS and Chroma.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np

class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    async def add(
        self,
        vectors: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """Add vectors and documents to the store."""
        pass
        
    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        k: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
        
    @abstractmethod
    async def clear(self) -> None:
        """Clear the vector store."""
        pass

class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store implementation."""
    
    def __init__(self, dimension: int = 1536):
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS is required. Install with: pip install faiss-cpu"
            )
            
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        
    async def add(
        self,
        vectors: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """Add vectors and documents to FAISS."""
        if len(vectors) != len(documents):
            raise ValueError("Number of vectors must match number of documents")
            
        # Convert to numpy array and add to index
        vectors_np = np.array(vectors).astype('float32')
        self.index.add(vectors_np)
        
        # Store documents and metadata
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
            
    async def search(
        self,
        query_vector: List[float],
        k: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in FAISS."""
        # Convert query to numpy array
        query_np = np.array([query_vector]).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_np, k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "distance": float(distances[0][i])
                })
                
        return results
        
    async def clear(self) -> None:
        """Clear the FAISS index and stored data."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadata = []

class ChromaVectorStore(BaseVectorStore):
    """Chroma-based vector store implementation."""
    
    def __init__(self, collection_name: str = "default"):
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB is required. Install with: pip install chromadb"
            )
            
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
    async def add(
        self,
        vectors: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """Add vectors and documents to Chroma."""
        if len(vectors) != len(documents):
            raise ValueError("Number of vectors must match number of documents")
            
        # Prepare IDs and metadata
        ids = [str(i) for i in range(len(documents))]
        if not metadata:
            metadata = [{}] * len(documents)
            
        # Add to collection
        self.collection.add(
            embeddings=vectors,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        
    async def search(
        self,
        query_vector: List[float],
        k: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Chroma."""
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k
        )
        
        # Prepare results in consistent format
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
            
        return formatted_results
        
    async def clear(self) -> None:
        """Clear the Chroma collection."""
        self.collection.delete()
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        ) 