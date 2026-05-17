"""
Main RAG implementation that orchestrates the modular components.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..document_loader import BaseDocumentLoader as DocumentLoader
from ..document_processing import DocumentProcessor
from ..document_processing.base import Document
from ..embeddings import EmbeddingConfig, EmbeddingGenerator
from ..vector_store import VectorStore, VectorStoreConfig


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
        # Initialize document processor after embedding generator so we can use it as model
        self.document_processor = self._get_document_processor()
        self.logger = logging.getLogger(__name__)

    def _get_retriever(self):
        """Get appropriate retriever with lazy import."""
        if self.retriever is None:
            # Lazy import to avoid circular dependency
            from ..retrieval import RetrievalConfig, Retriever

            # Create RetrievalConfig from the dict
            retrieval_config = RetrievalConfig(
                vector_store=self.vector_store,
                document_processor=self.document_processor,
                embedding_generator=self.embedding_generator,
                top_k=self.config.retrieval_config.get("top_k", 5),
                similarity_threshold=self.config.retrieval_config.get("similarity_threshold", 0.7),
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

        # Extract api_key separately to avoid duplicate keyword argument
        custom_params = (cfg.custom_params or {}).copy()
        api_key = custom_params.pop("api_key", None)

        return EmbeddingModel(
            model_type=model_type, model_name=cfg.model_name, api_key=api_key, **custom_params
        )

    def _get_document_loader(self) -> DocumentLoader:
        """Get appropriate document loader."""
        # Use LocalDocumentLoader as default, can be extended for other sources
        from ..document_loader.document_loader import LocalDocumentLoader

        return LocalDocumentLoader(**self.config.document_config)

    def _get_document_processor(self) -> DocumentProcessor:
        """Get appropriate document processor."""
        # Use EnhancedDocumentProcessor as default
        from ..document_processing.document_processor import (
            EnhancedDocumentProcessor,
            ProcessingConfig,
        )
        from ..models.base import BaseLLM

        # Create a wrapper that makes embedding_generator compatible with semantic chunker
        # The chunker expects model.embeddings() but embedding_generator has generate() or generate_batch_embeddings()
        class EmbeddingModelWrapper(BaseLLM):
            """Wrapper to make embedding generator work as a model for document processing."""

            def __init__(self, embedding_generator):
                super().__init__("embedding_wrapper")
                self.embedding_generator = embedding_generator

            async def embeddings(self, texts):
                """Generate embeddings for texts - compatible with semantic chunker."""
                if isinstance(texts, str):
                    texts = [texts]

                # Try different methods the embedding generator might have
                if hasattr(self.embedding_generator, "generate_batch_embeddings"):
                    return await self.embedding_generator.generate_batch_embeddings(texts)
                elif hasattr(self.embedding_generator, "generate"):
                    # generate() takes a list of texts and returns a list of embeddings
                    return await self.embedding_generator.generate(texts)
                else:
                    # Fallback: return empty embeddings
                    return [[0.0] * 384 for _ in texts]

            # Implement required abstract methods from BaseLLM (stubs - not used by document processor)
            async def generate(
                self,
                prompt: str,
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                **kwargs,
            ) -> str:
                raise NotImplementedError(
                    "This wrapper is only for embeddings, not text generation"
                )

            async def generate_stream(
                self,
                prompt: str,
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                **kwargs,
            ):
                raise NotImplementedError(
                    "This wrapper is only for embeddings, not text generation"
                )

            async def chat(
                self,
                messages: List[Dict[str, str]],
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                **kwargs,
            ) -> str:
                raise NotImplementedError(
                    "This wrapper is only for embeddings, not text generation"
                )

            async def chat_stream(
                self,
                messages: List[Dict[str, str]],
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                **kwargs,
            ):
                raise NotImplementedError(
                    "This wrapper is only for embeddings, not text generation"
                )

        # Use wrapper if embedding generator is available, otherwise None
        model_for_processor = (
            EmbeddingModelWrapper(self.embedding_generator) if self.embedding_generator else None
        )

        return EnhancedDocumentProcessor(
            model=model_for_processor, config=ProcessingConfig(**self.config.document_config)
        )

    async def initialize(self) -> None:
        """Initialize all components."""
        await self.vector_store.initialize()
        retriever = self._get_retriever()
        await retriever.initialize()
        # Initialize embedding generator if it has an initialize method
        if hasattr(self.embedding_generator, "initialize"):
            await self.embedding_generator.initialize()

    async def add_documents(self, documents: List[Document], process: bool = True) -> None:
        """Add documents to the RAG system."""
        if process:
            # Check if document processor has a model (required for semantic chunking)
            has_model = (
                hasattr(self.document_processor, "model")
                and self.document_processor.model is not None
            )

            # Process documents if processor supports it and has a model
            if has_model and hasattr(self.document_processor, "process_batch"):
                documents = await self.document_processor.process_batch(documents)
            elif has_model and hasattr(self.document_processor, "process_documents"):
                # Convert Document objects to text strings for processing
                original_docs = documents  # Save original for source reference
                texts = [doc.content for doc in documents]
                metadata_list = [doc.metadata for doc in documents]
                processed_chunks = await self.document_processor.process_documents(
                    texts, metadata_list
                )
                # Flatten the list of lists and convert back to Document objects
                documents = []
                for doc_idx, chunks in enumerate(processed_chunks):
                    for chunk_idx, chunk in enumerate(chunks):
                        # Handle both dict and object chunks
                        if isinstance(chunk, dict):
                            chunk_text = chunk.get("text", "")
                            chunk_metadata = chunk.get("metadata", {})
                        else:
                            chunk_text = getattr(chunk, "text", str(chunk))
                            chunk_metadata = getattr(chunk, "metadata", {})

                        source = (
                            original_docs[doc_idx].source
                            if doc_idx < len(original_docs)
                            else "unknown"
                        )
                        # Document from base.py requires: id, content, metadata, source
                        # But it's a dataclass, so we need to check the actual structure
                        documents.append(
                            Document(
                                id=f"doc_{doc_idx}_chunk_{chunk_idx}",
                                content=chunk_text,
                                metadata=chunk_metadata,
                                source=source,
                            )
                        )
            # If no model or processing method available, use documents as-is (no chunking)

        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = await self.embedding_generator.generate(texts)

        # Add to vector store
        metadatas = [doc.metadata for doc in documents]
        docs = [{"content": doc.content} for doc in documents]
        await self.vector_store.add_vectors(embeddings, metadatas, docs)

    async def retrieve(
        self, query: str, k: int = 5, filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve relevant documents."""
        retriever = self._get_retriever()
        # Retriever.retrieve() expects top_k as keyword arg and filter_criteria in **kwargs
        kwargs = {}
        if filter_criteria:
            kwargs.update(filter_criteria)
        return await retriever.retrieve(query, top_k=k, **kwargs)

    async def clear(self) -> None:
        """Clear all documents from the system."""
        await self.vector_store.clear()
        if self.retriever:
            await self.retriever.clear()
