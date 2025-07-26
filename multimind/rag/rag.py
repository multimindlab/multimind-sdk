"""
Main RAG implementation that orchestrates the modular components.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..vector_store import VectorStore, VectorStoreConfig
from ..document_processing import DocumentProcessor, Document
from ..document_loader import BaseDocumentLoader as DocumentLoader
from ..embeddings import EmbeddingGenerator, EmbeddingConfig

@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    vector_store_config: VectorStoreConfig
    retrieval_config: Dict[str, Any]  # Changed from RetrievalConfig to avoid circular import
    embedding_config: EmbeddingConfig
    document_config: Dict[str, Any]
    custom_params: Dict[str, Any] = None

class RAG:
    """RAG system that orchestrates the modular components."""
    
    def __init__(self, config: RAGConfig):
        """
        Initialize RAG system.
        
        Args:
            config: RAG configuration
        """
        self.config = config
        self.vector_store = VectorStore(config.vector_store_config)
        self.retriever = None  # Will be initialized lazily
        self.embedding_generator = self._get_embedding_generator()
        self.document_loader = self._get_document_loader()
        self.document_processor = self._get_document_processor()
        self.logger = logging.getLogger(__name__)

    def _get_retriever(self):
        """Get appropriate retriever with lazy import."""
        if self.retriever is None:
            # Lazy import to avoid circular dependency
            from ..retrieval import Retriever, RetrievalConfig
            
            # Create RetrievalConfig from the dict
            retrieval_config = RetrievalConfig(
                vector_store=self.vector_store,
                document_processor=self.document_processor,
                embedding_generator=self.embedding_generator,
                top_k=self.config.retrieval_config.get('top_k', 5),
                similarity_threshold=self.config.retrieval_config.get('similarity_threshold', 0.7)
            )
            
            self.retriever = Retriever(retrieval_config)
        return self.retriever

    def _get_embedding_generator(self) -> EmbeddingGenerator:
        """Get appropriate embedding generator."""
        # Use EmbeddingModel from multimind/embeddings/embedding.py
        from ..embeddings.embedding import EmbeddingModel, EmbeddingType
        cfg = self.config.embedding_config
        # Assume cfg has model_type as string, convert to EmbeddingType
        model_type = EmbeddingType(cfg.model_type)
        return EmbeddingModel(
            model_type=model_type,
            model_name=cfg.model_name,
            api_key=cfg.custom_params.get('api_key') if cfg.custom_params else None,
            **(cfg.custom_params or {})
        )

    def _get_document_loader(self) -> DocumentLoader:
        """Get appropriate document loader."""
        # Use LocalDocumentLoader as default, can be extended for other sources
        from ..document_loader.document_loader import LocalDocumentLoader
        return LocalDocumentLoader(**self.config.document_config)

    def _get_document_processor(self) -> DocumentProcessor:
        """Get appropriate document processor."""
        # Use EnhancedDocumentProcessor as default
        from ..document_processing.document_processor import EnhancedDocumentProcessor, ProcessingConfig
        # If a model is needed, pass None or a default
        return EnhancedDocumentProcessor(
            model=None,
            config=ProcessingConfig(**self.config.document_config)
        )

    async def initialize(self) -> None:
        """Initialize all components."""
        await self.vector_store.initialize()
        retriever = self._get_retriever()
        await retriever.initialize()
        await self.embedding_generator.initialize()

    async def add_documents(
        self,
        documents: List[Document],
        process: bool = True
    ) -> None:
        """Add documents to the RAG system."""
        if process:
            documents = await self.document_processor.process_batch(documents)
        
        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = await self.embedding_generator.generate(texts)
        
        # Add to vector store
        metadatas = [doc.metadata for doc in documents]
        docs = [{"content": doc.content} for doc in documents]
        await self.vector_store.add_vectors(embeddings, metadatas, docs)

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve relevant documents."""
        retriever = self._get_retriever()
        return await retriever.retrieve(query, k, filter_criteria)

    async def clear(self) -> None:
        """Clear all documents from the system."""
        await self.vector_store.clear()
        if self.retriever:
            await self.retriever.clear()