"""
Base retriever implementation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..core.exceptions import RetrievalError
from ..vector_store import VectorStore
from ..document_processing import DocumentProcessor
from ..embeddings import EmbeddingGenerator

@dataclass
class RetrievalResult:
    """Represents a retrieval result with metadata."""
    content: str
    score: float
    metadata: Dict[str, Any]
    document_id: Optional[str] = None
    source: Optional[str] = None
    chunk_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate retrieval result after initialization."""
        if not isinstance(self.content, str):
            raise ValueError("Retrieval result content must be a string")
        if not isinstance(self.score, (int, float)):
            raise ValueError("Retrieval result score must be a number")
        if not isinstance(self.metadata, dict):
            raise ValueError("Retrieval result metadata must be a dictionary")

@dataclass
class RetrievalConfig:
    """Configuration for retriever."""
    vector_store: VectorStore
    document_processor: DocumentProcessor
    embedding_generator: EmbeddingGenerator
    top_k: int = 5
    similarity_threshold: float = 0.7

class Retriever:
    """Base retriever implementation."""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.vector_store = config.vector_store
        self.document_processor = config.document_processor
        self.embedding_generator = config.embedding_generator

    async def initialize(self) -> None:
        """Initialize the retriever."""
        pass

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve documents for a query."""
        try:
            # Generate embedding for query
            query_embedding = await self.embedding_generator.generate_embedding(query)
            
            # Search vector store
            results = await self.vector_store.search(
                query_embedding,
                top_k=top_k or self.config.top_k,
                **kwargs
            )
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for result in results:
                if result.get("score", 0) >= self.config.similarity_threshold:
                    retrieval_results.append(RetrievalResult(
                        content=result.get("content", ""),
                        score=result.get("score", 0.0),
                        metadata=result.get("metadata", {}),
                        document_id=result.get("document_id"),
                        source=result.get("source"),
                        chunk_id=result.get("chunk_id")
                    ))
            
            return retrieval_results
            
        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {str(e)}")

    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> None:
        """Add documents to the retriever."""
        try:
            # Process documents
            processed_docs = []
            for doc in documents:
                processed_doc = await self.document_processor.process(doc)
                processed_docs.append(processed_doc)
            
            # Generate embeddings
            embeddings = []
            for doc in processed_docs:
                embedding = await self.embedding_generator.generate_embedding(
                    doc.get("content", "")
                )
                embeddings.append(embedding)
            
            # Add to vector store
            await self.vector_store.add_documents(
                processed_docs,
                embeddings,
                **kwargs
            )
            
        except Exception as e:
            raise RetrievalError(f"Failed to add documents: {str(e)}")

    async def delete_documents(
        self,
        document_ids: List[str],
        **kwargs
    ) -> None:
        """Delete documents from the retriever."""
        try:
            await self.vector_store.delete_documents(document_ids, **kwargs)
        except Exception as e:
            raise RetrievalError(f"Failed to delete documents: {str(e)}")

    async def update_documents(
        self,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> None:
        """Update documents in the retriever."""
        try:
            # Process documents
            processed_docs = []
            for doc in documents:
                processed_doc = await self.document_processor.process(doc)
                processed_docs.append(processed_doc)
            
            # Generate embeddings
            embeddings = []
            for doc in processed_docs:
                embedding = await self.embedding_generator.generate_embedding(
                    doc.get("content", "")
                )
                embeddings.append(embedding)
            
            # Update in vector store
            await self.vector_store.update_documents(
                processed_docs,
                embeddings,
                **kwargs
            )
            
        except Exception as e:
            raise RetrievalError(f"Failed to update documents: {str(e)}")

    async def clear(self) -> None:
        """Clear all documents from the retriever."""
        try:
            await self.vector_store.clear()
        except Exception as e:
            raise RetrievalError(f"Failed to clear documents: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "vector_store_stats": self.vector_store.get_stats(),
            "document_processor_stats": self.document_processor.get_stats(),
            "embedding_generator_stats": self.embedding_generator.get_stats(),
            "config": {
                "top_k": self.config.top_k,
                "similarity_threshold": self.config.similarity_threshold
            }
        } 