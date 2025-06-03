"""
Hybrid workflow system for RAG and vision+language tasks.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel
from datetime import datetime
import asyncio
from ..core.provider import ProviderAdapter, GenerationResult, EmbeddingResult, ImageAnalysisResult
from ..core.router import Router, TaskType

class Document(BaseModel):
    """Represents a document in the RAG system."""
    content: str
    metadata: Dict[str, Any] = {}
    embeddings: Optional[List[float]] = None
    created_at: datetime = datetime.now()

class ImageDocument(BaseModel):
    """Represents an image document in the RAG system."""
    image_data: bytes
    metadata: Dict[str, Any] = {}
    analysis: Optional[Dict[str, Any]] = None
    created_at: datetime = datetime.now()

class SharedContext(BaseModel):
    """Shared context that can be used across providers."""
    documents: List[Document] = []
    image_documents: List[ImageDocument] = []
    embeddings: Optional[List[float]] = None
    image_analysis: Optional[Dict[str, Any]] = None
    text_context: Optional[str] = None
    metadata: Dict[str, Any] = {}

class HybridWorkflow:
    """Manages hybrid RAG and vision+language workflows."""
    
    def __init__(self, router: Router):
        """Initialize the hybrid workflow manager."""
        self.router = router
        self.shared_contexts: Dict[str, SharedContext] = {}
    
    async def add_document(
        self,
        content: str,
        context_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        generate_embeddings: bool = True
    ) -> None:
        """Add a document to the shared context."""
        context = self.shared_contexts.get(context_id)
        if not context:
            context = SharedContext()
            self.shared_contexts[context_id] = context
        
        document = Document(
            content=content,
            metadata=metadata or {}
        )
        
        if generate_embeddings:
            embedding_result = await self.router.route(
                TaskType.EMBEDDINGS,
                content,
                model="text-embedding-ada-002"
            )
            document.embeddings = embedding_result.embeddings
        
        context.documents.append(document)
    
    async def add_image_document(
        self,
        image_data: bytes,
        context_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        analyze_image: bool = True
    ) -> None:
        """Add an image document to the shared context."""
        context = self.shared_contexts.get(context_id)
        if not context:
            context = SharedContext()
            self.shared_contexts[context_id] = context
        
        image_doc = ImageDocument(
            image_data=image_data,
            metadata=metadata or {}
        )
        
        if analyze_image:
            analysis_result = await self.router.route(
                TaskType.IMAGE_ANALYSIS,
                image_data,
                prompt="Analyze this image in detail."
            )
            image_doc.analysis = {
                "objects": analysis_result.objects,
                "captions": analysis_result.captions,
                "text": analysis_result.text
            }
        
        context.image_documents.append(image_doc)
    
    async def process_with_rag(
        self,
        query: str,
        context_id: str,
        task_type: TaskType = TaskType.TEXT_GENERATION,
        **kwargs
    ) -> GenerationResult:
        """Process a query using RAG with shared embeddings."""
        context = self.shared_contexts.get(context_id)
        if not context:
            raise ValueError(f"No context found for ID: {context_id}")
        
        # Generate query embeddings
        query_embedding_result = await self.router.route(
            TaskType.EMBEDDINGS,
            query,
            model="text-embedding-ada-002"
        )
        query_embeddings = query_embedding_result.embeddings
        
        # Find relevant documents
        relevant_docs = await self._find_relevant_documents(
            query_embeddings,
            context.documents,
            top_k=kwargs.pop("top_k", 3)
        )
        
        # Generate response using relevant documents
        result = await self.router.route(
            task_type,
            {
                "query": query,
                "context": "\n".join(doc.content for doc in relevant_docs),
                "embeddings": query_embeddings,
                "metadata": {
                    "relevant_docs": [doc.metadata for doc in relevant_docs]
                }
            },
            **kwargs
        )
        
        return result
    
    async def process_vision_language(
        self,
        image_data: bytes,
        prompt: str,
        context_id: str,
        **kwargs
    ) -> GenerationResult:
        """Process an image and text prompt using shared context."""
        context = self.shared_contexts.get(context_id)
        if not context:
            raise ValueError(f"No context found for ID: {context_id}")
        
        # Find relevant image documents
        relevant_images = await self._find_relevant_images(
            prompt,
            context.image_documents,
            top_k=kwargs.pop("top_k", 1)
        )
        
        # Generate response using relevant images and context
        result = await self.router.route(
            TaskType.TEXT_GENERATION,
            {
                "prompt": prompt,
                "image_analysis": [img.analysis for img in relevant_images],
                "context": context.text_context,
                "metadata": {
                    "relevant_images": [img.metadata for img in relevant_images]
                }
            },
            **kwargs
        )
        
        return result
    
    async def process_hybrid(
        self,
        query: str,
        image_data: Optional[bytes] = None,
        context_id: str = "default",
        **kwargs
    ) -> GenerationResult:
        """Process a hybrid query that may include both text and image."""
        context = self.shared_contexts.get(context_id)
        if not context:
            raise ValueError(f"No context found for ID: {context_id}")
        
        # Process image if provided
        image_analysis = None
        if image_data:
            analysis_result = await self.router.route(
                TaskType.IMAGE_ANALYSIS,
                image_data,
                prompt=query
            )
            image_analysis = {
                "objects": analysis_result.objects,
                "captions": analysis_result.captions,
                "text": analysis_result.text
            }
        
        # Generate query embeddings
        query_embedding_result = await self.router.route(
            TaskType.EMBEDDINGS,
            query,
            model="text-embedding-ada-002"
        )
        query_embeddings = query_embedding_result.embeddings
        
        # Find relevant documents and images
        relevant_docs = await self._find_relevant_documents(
            query_embeddings,
            context.documents,
            top_k=kwargs.pop("top_k", 3)
        )
        
        relevant_images = await self._find_relevant_images(
            query,
            context.image_documents,
            top_k=kwargs.pop("top_k", 1)
        )
        
        # Generate response using all context
        result = await self.router.route(
            TaskType.TEXT_GENERATION,
            {
                "query": query,
                "embeddings": query_embeddings,
                "image_analysis": image_analysis or [img.analysis for img in relevant_images],
                "context": "\n".join(doc.content for doc in relevant_docs),
                "metadata": {
                    "relevant_docs": [doc.metadata for doc in relevant_docs],
                    "relevant_images": [img.metadata for img in relevant_images]
                }
            },
            **kwargs
        )
        
        return result
    
    async def _find_relevant_documents(
        self,
        query_embeddings: List[float],
        documents: List[Document],
        top_k: int = 3
    ) -> List[Document]:
        """Find the most relevant documents using cosine similarity."""
        if not documents:
            return []
        
        # Calculate cosine similarity for each document
        similarities = []
        for doc in documents:
            if doc.embeddings:
                similarity = self._cosine_similarity(query_embeddings, doc.embeddings)
                similarities.append((doc, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in similarities[:top_k]]
    
    async def _find_relevant_images(
        self,
        query: str,
        images: List[ImageDocument],
        top_k: int = 1
    ) -> List[ImageDocument]:
        """Find the most relevant images using semantic similarity."""
        if not images:
            return []
        
        # Generate query embeddings
        query_embedding_result = await self.router.route(
            TaskType.EMBEDDINGS,
            query,
            model="text-embedding-ada-002"
        )
        query_embeddings = query_embedding_result.embeddings
        
        # Calculate similarity for each image's analysis
        similarities = []
        for img in images:
            if img.analysis and img.analysis.get("text"):
                # Generate embeddings for image analysis text
                analysis_embedding_result = await self.router.route(
                    TaskType.EMBEDDINGS,
                    img.analysis["text"],
                    model="text-embedding-ada-002"
                )
                similarity = self._cosine_similarity(
                    query_embeddings,
                    analysis_embedding_result.embeddings
                )
                similarities.append((img, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [img for img, _ in similarities[:top_k]]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def get_context(self, context_id: str) -> Optional[SharedContext]:
        """Get the shared context for a given ID."""
        return self.shared_contexts.get(context_id)
    
    def clear_context(self, context_id: str) -> None:
        """Clear the shared context for a given ID."""
        if context_id in self.shared_contexts:
            del self.shared_contexts[context_id] 