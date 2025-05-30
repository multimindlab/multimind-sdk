"""
Vector store implementations using FAISS, Chroma, and Pinecone.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image

class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def add(
        self,
        vectors: List[List[float]],
        documents: List[Union[str, Image.Image]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """Add vectors and documents to the store, supporting both text and image inputs."""
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        k: int = 3,
        input_type: str = "text",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for the top-k similar vectors, supporting multi-modal queries."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear the vector store."""
        pass

    @abstractmethod
    async def get_document_count(self) -> int:
        """Get the total number of documents in the store."""
        pass

    @abstractmethod
    async def hybrid_search(
        self,
        query_vector: List[float],
        keywords: Optional[str] = None,
        k: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector similarity and keyword matching."""
        pass

    @abstractmethod
    async def update_vector(
        self,
        document_id: str,
        new_vector: List[float],
        **kwargs
    ) -> None:
        """Update an existing vector in the store."""
        pass

class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store implementation with multi-modal support."""

    def __init__(self, dimension: int = 1536):
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS is required. Install with: pip install faiss-cpu"
            )

        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[Union[str, Image.Image]] = []
        self.metadata: List[Dict[str, Any]] = []

    async def add(
        self,
        vectors: List[List[float]],
        documents: List[Union[str, Image.Image]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """Add vectors and documents to the FAISS index."""
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
        input_type: str = "text",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for the top-k similar vectors in the FAISS index."""
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

    async def get_document_count(self) -> int:
        """Get the total number of documents in the store."""
        return len(self.documents)

    async def hybrid_search(
        self,
        query_vector: List[float],
        keywords: Optional[str] = None,
        k: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector similarity and keyword matching."""
        # For FAISS, we will first perform a vector search, then filter by keywords
        vector_results = await self.search(query_vector, k, **kwargs)

        if not keywords:
            return vector_results

        # Filter results by keywords (if any)
        keyword_filtered_results = [
            result for result in vector_results
            if keywords.lower() in result["document"].lower()
        ]

        return keyword_filtered_results[:k]  # Return top-k results

    async def update_vector(
        self,
        document_id: str,
        new_vector: List[float],
        **kwargs
    ) -> None:
        """Update an existing vector in FAISS."""
        # FAISS does not support in-place updates, so we need to remove and re-add
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS is required. Install with: pip install faiss-cpu"
            )

        # Find the index of the document
        index_to_update = None
        for i, doc in enumerate(self.documents):
            if doc.get("id") == document_id:
                index_to_update = i
                break

        if index_to_update is None:
            raise ValueError(f"Document ID {document_id} not found")

        # Remove the old vector
        self.index.remove_ids(np.array([index_to_update]).astype('int64'))

        # Add the new vector
        self.index.add(np.array([new_vector]).astype('float32'))

        # Update the document and metadata
        self.documents[index_to_update] = document_id
        self.metadata[index_to_update] = kwargs.get("metadata", {})

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

        # Prepare results in consistent forma
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

    async def get_document_count(self) -> int:
        """Get the total number of documents in the store."""
        return self.collection.count()  # ChromaDB's collection.count() returns number of documents

    async def hybrid_search(
        self,
        query_vector: List[float],
        keywords: Optional[str] = None,
        k: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector similarity and keyword matching."""
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k
        )

        # Prepare results in consistent forma
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        if not keywords:
            return formatted_results

        # Filter results by keywords (if any)
        keyword_filtered_results = [
            result for result in formatted_results
            if keywords.lower() in result["document"].lower()
        ]

        return keyword_filtered_results[:k]  # Return top-k results

    async def update_vector(
        self,
        document_id: str,
        new_vector: List[float],
        **kwargs
    ) -> None:
        """Update an existing vector in Chroma."""
        # ChromaDB does not support in-place updates, so we need to remove and re-add
        self.collection.delete(ids=[document_id])
        self.collection.add(
            embeddings=[new_vector],
            documents=[document_id],
            metadatas=[kwargs.get("metadata", {})],
            ids=[document_id]
        )

class PineconeVectorStore(BaseVectorStore):
    """Pinecone vector store implementation."""

    def __init__(self, api_key: str, environment: str, index_name: str):
        try:
            import pinecone
        except ImportError:
            raise ImportError(
                "Pinecone is required. Install with: pip install pinecone-client"
            )

        # Initialize Pinecone client
        pinecone.init(
            api_key=api_key,
            environment=environment
        )
        self.index_name = index_name
        self.index = pinecone.Index(index_name)

    async def add(
        self,
        vectors: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """Add vectors and documents to Pinecone."""
        if len(vectors) != len(documents):
            raise ValueError("Number of vectors must match number of documents")

        # Prepare items for upsert
        upsert_items = [
            (str(i), vectors[i], documents[i], metadata[i] if metadata else {})
            for i in range(len(documents))
        ]

        # Upsert to Pinecone
        self.index.upsert(
            vectors=upsert_items
        )

    async def search(
        self,
        query_vector: List[float],
        k: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone."""
        # Query Pinecone index
        results = self.index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True
        )

        # Prepare results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "document": match.metadata.get("document", ""),
                "metadata": match.metadata,
                "distance": match.score
            })

        return formatted_results

    async def clear(self) -> None:
        """Clear the Pinecone index."""
        self.index.delete(delete_all=True)

    async def get_document_count(self) -> int:
        """Get the total number of documents in the store."""
        return self.index.describe_index_stats().total_vector_count

    async def hybrid_search(
        self,
        query_vector: List[float],
        keywords: Optional[str] = None,
        k: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector similarity and keyword matching."""
        # Query Pinecone index
        results = self.index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True
        )

        # Prepare results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "document": match.metadata.get("document", ""),
                "metadata": match.metadata,
                "distance": match.score
            })

        if not keywords:
            return formatted_results

        # Filter results by keywords (if any)
        keyword_filtered_results = [
            result for result in formatted_results
            if keywords.lower() in result["document"].lower()
        ]

        return keyword_filtered_results[:k]  # Return top-k results

    async def update_vector(
        self,
        document_id: str,
        new_vector: List[float],
        **kwargs
    ) -> None:
        """Update an existing vector in Pinecone."""
        # Prepare the item for upsert
        upsert_item = (document_id, new_vector, kwargs.get("metadata", {}))

        # Upsert to Pinecone
        self.index.upsert(
            vectors=[upsert_item]
        )