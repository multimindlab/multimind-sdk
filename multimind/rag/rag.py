"""
Concrete RAG implementation.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import asyncio
from .base import BaseRAG
from .vector_store import BaseVectorStore, FAISSVectorStore, ChromaVectorStore
from .embeddings import get_embedder, BaseLLM
from .document import Document, DocumentProcessor
from ..models.base import BaseLLM as BaseModel

class RAG(BaseRAG):
    """Concrete RAG implementation."""
    
    def __init__(
        self,
        embedder: Union[str, BaseLLM],
        vector_store: Optional[Union[str, BaseVectorStore]] = None,
        model: Optional[BaseModel] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 3,
        **kwargs
    ):
        """Initialize RAG system.
        
        Args:
            embedder: Embedder type or instance
            vector_store: Vector store type or instance (default: FAISS)
            model: LLM for generating responses (optional)
            chunk_size: Maximum size of text chunks in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            top_k: Number of top documents to retrieve
            **kwargs: Additional arguments for components
        """
        # Initialize embedder
        if isinstance(embedder, str):
            self.embedder = get_embedder(embedder, **kwargs)
        else:
            self.embedder = embedder
            
        # Initialize vector store
        if vector_store is None:
            self.vector_store = FAISSVectorStore()
        elif isinstance(vector_store, str):
            if vector_store == "faiss":
                self.vector_store = FAISSVectorStore(**kwargs)
            elif vector_store == "chroma":
                self.vector_store = ChromaVectorStore(**kwargs)
            else:
                raise ValueError(
                    f"Unsupported vector store type: {vector_store}. "
                    "Supported types: faiss, chroma"
                )
        else:
            self.vector_store = vector_store
            
        # Initialize document processor
        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
        
        # Store other settings
        self.model = model
        self.top_k = top_k
        
    async def add_documents(
        self,
        documents: Union[str, Document, List[Union[str, Document]]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add documents to the RAG system.
        
        Args:
            documents: Document(s) to add
            metadata: Optional metadata to add to documents
        """
        # Convert to list
        if not isinstance(documents, list):
            documents = [documents]
            
        # Process documents
        processed_docs = []
        for doc in documents:
            processed_docs.extend(
                self.processor.process_document(doc, metadata)
            )
            
        # Generate embeddings
        texts = [doc.text for doc in processed_docs]
        embeddings = await self.embedder.embed(texts)
        
        # Add to vector store
        await self.vector_store.add_documents(
            documents=processed_docs,
            embeddings=embeddings
        )
        
    async def add_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a file to the RAG system.
        
        Args:
            file_path: Path to file
            metadata: Optional metadata to add to documents
        """
        # Process file
        documents = self.processor.process_file(file_path, metadata)
        
        # Add documents
        await self.add_documents(documents)
        
    async def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """Query the RAG system.
        
        Args:
            query: Query text
            top_k: Number of top documents to retrieve (default: self.top_k)
            **kwargs: Additional arguments for search
            
        Returns:
            List of (document, score) tuples
        """
        # Generate query embedding
        query_embedding = (await self.embedder.embed([query]))[0]
        
        # Search vector store
        results = await self.vector_store.search(
            query_embedding,
            top_k=top_k or self.top_k,
            **kwargs
        )
        
        return results
        
    async def generate(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a response using the RAG system.
        
        Args:
            query: Query text
            top_k: Number of top documents to retrieve
            **kwargs: Additional arguments for model generation
            
        Returns:
            Generated response
            
        Raises:
            ValueError: If no model is set
        """
        if not self.model:
            raise ValueError("No model set for generation")
            
        # Retrieve relevant documents
        results = await self.query(query, top_k=top_k)
        
        # Format context
        context = "\n\n".join(
            f"Document {i+1}:\n{doc.text}"
            for i, (doc, _) in enumerate(results)
        )
        
        # Generate prompt
        prompt = f"""Use the following context to answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response
        response = await self.model.generate(prompt, **kwargs)
        
        return response
        
    async def clear(self) -> None:
        """Clear all documents from the RAG system."""
        await self.vector_store.clear()
        
    async def get_document_count(self) -> int:
        """Get the number of documents in the RAG system."""
        return await self.vector_store.get_document_count()
        
    async def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        # Generate a test embedding
        test_embedding = (await self.embedder.embed(["test"]))[0]
        return len(test_embedding)
        
    @classmethod
    async def from_documents(
        cls,
        documents: List[Union[str, Document]],
        embedder: Union[str, BaseLLM],
        vector_store: Optional[Union[str, BaseVectorStore]] = None,
        model: Optional[BaseModel] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "RAG":
        """Create a RAG instance from a list of documents.
        
        Args:
            documents: List of documents to add
            embedder: Embedder type or instance
            vector_store: Vector store type or instance
            model: LLM for generating responses
            metadata: Optional metadata to add to documents
            **kwargs: Additional arguments for RAG initialization
            
        Returns:
            Initialized RAG instance
        """
        # Create instance
        rag = cls(
            embedder=embedder,
            vector_store=vector_store,
            model=model,
            **kwargs
        )
        
        # Add documents
        await rag.add_documents(documents, metadata)
        
        return rag
        
    @classmethod
    async def from_files(
        cls,
        file_paths: List[Union[str, Path]],
        embedder: Union[str, BaseLLM],
        vector_store: Optional[Union[str, BaseVectorStore]] = None,
        model: Optional[BaseModel] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "RAG":
        """Create a RAG instance from a list of files.
        
        Args:
            file_paths: List of file paths to add
            embedder: Embedder type or instance
            vector_store: Vector store type or instance
            model: LLM for generating responses
            metadata: Optional metadata to add to documents
            **kwargs: Additional arguments for RAG initialization
            
        Returns:
            Initialized RAG instance
        """
        # Create instance
        rag = cls(
            embedder=embedder,
            vector_store=vector_store,
            model=model,
            **kwargs
        )
        
        # Add files
        for file_path in file_paths:
            await rag.add_file(file_path, metadata)
            
        return rag 