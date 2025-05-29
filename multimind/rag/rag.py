"""
Concrete RAG implementation.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, cast, Sequence
from pathlib import Path
import asyncio
from .base import BaseRAG
from .vector_store import BaseVectorStore, FAISSVectorStore, ChromaVectorStore
from .embeddings import get_embedder, BaseLLM, ImageEmbedder
from .document import Document, DocumentProcessor
from ..models.base import BaseLLM as BaseModel
from PIL import Image

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
        """Initialize RAG system."""
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

        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        super().__init__(embedder=self.embedder, vector_store=self.vector_store, **kwargs)

    def _ensure_list_of_vectors(self, embeddings: Union[List[float], List[List[float]]]) -> List[List[float]]:
        """Convert embeddings to List[List[float]] format."""
        if isinstance(embeddings, list):
            if all(isinstance(x, float) for x in embeddings):
                return [cast(List[float], embeddings)]  # Single embedding
            elif all(isinstance(x, list) and all(isinstance(y, float) for y in x) for x in embeddings):
                return cast(List[List[float]], embeddings)  # List of embeddings
        raise ValueError("Invalid embeddings format. Expected List[float] or List[List[float]]")

    async def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """Add documents to the vector store."""
        # Process documents
        processed_docs = []
        processed_texts = []
        for i, doc in enumerate(documents):
            doc_metadata = metadata[i] if metadata else {}
            chunks = self.processor.process_document(doc, doc_metadata)
            processed_docs.extend(chunks)
            processed_texts.extend([chunk.text for chunk in chunks])

        # Generate embeddings
        raw_embeddings = await self.embedder.embeddings(processed_texts)
        embeddings = self._ensure_list_of_vectors(raw_embeddings)

        # Add to vector store
        await self.vector_store.add(
            vectors=embeddings,
            documents=processed_texts,
            metadata=[doc.metadata for doc in processed_docs]
        )

    async def search(
        self,
        query: str,
        k: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        # Generate query embedding
        raw_query_embedding = await self.embedder.embeddings(query)
        if isinstance(raw_query_embedding[0], list):
            query_embedding = cast(List[float], raw_query_embedding[0])
        else:
            query_embedding = cast(List[float], raw_query_embedding)

        # Search vector store
        results = await self.vector_store.search(query_vector=query_embedding, k=k)
        return results

    async def query(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> str:
        """Query the RAG system with optional context."""
        if context is None:
            # Search for relevant documents
            context = await self.search(query, k=self.top_k)

        if self.model:
            # Format context for the model
            context_text = "\n\n".join([doc["text"] for doc in context])
            prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
            
            # Generate response
            response = await self.model.generate(prompt, **kwargs)
            return response
        else:
            # If no model is set, return the context
            return "\n\n".join([doc["text"] for doc in context])

    def format_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Format the prompt using a template."""
        context_text = "\n\n".join([doc["text"] for doc in context])
        return f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

    async def evaluate(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the quality of results for a given query."""
        # Implement evaluation logic here
        pass

    async def parse_output(self, response: str) -> Union[str, Dict[str, Any]]:
        """Parse the output from the LLM into structured data or text."""
        # Implement output parsing logic here
        pass

    async def clear(self) -> None:
        """Clear all documents from the vector store."""
        # Forward to the underlying vector store's clear method
        await self.vector_store.clear()

    async def context_aware_retrieval(
        self,
        query: Union[str, Image.Image],
        user_history: Optional[List[Dict[str, Any]]] = None,
        input_type: str = "text",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve documents considering user history or session data, supporting image inputs."""
        if input_type == "image":
            # Generate image embeddings
            embedder = ImageEmbedder()
            query_embedding = embedder.embed([query])[0]
        else:
            # Generate text embeddings
            embedder = get_embedder("openai")
            query_embedding = await embedder.embed([query])[0]

        # Perform retrieval
        results = await self.vector_store.search(
            query_vector=query_embedding,
            k=kwargs.get("k", 3),
            input_type=input_type
        )

        return results

    def dynamic_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate a dynamic prompt based on query complexity."""
        if len(query.split()) > 10:  # Example: Complex query
            return f"Detailed Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        else:
            return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    async def multi_llm_orchestration(
        self,
        query: str,
        llms: List[BaseLLM],
        **kwargs
    ) -> str:
        """Orchestrate multiple LLMs with fallback and voting mechanisms."""
        responses = [await llm.generate(query, **kwargs) for llm in llms]
        # Implement voting or fallback logic here
        return responses[0]  # Example: Return the first response