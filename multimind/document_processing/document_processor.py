"""
Enhanced document processing with semantic chunking and metadata extraction.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import re
from dataclasses import dataclass
from enum import Enum
# Optional spacy import for NLP features
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spacy not available. NLP features will be disabled.")

# Optional beautifulsoup import for HTML processing
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("Warning: beautifulsoup4 not available. HTML processing features will be disabled.")

import requests

# Optional transformers import for advanced document processing
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _AUTO_MODEL_CLASS = AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
        _AUTO_MODEL_CLASS = AutoModelForSeq2SeqGeneration
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        try:
            from transformers import AutoTokenizer
            _AUTO_MODEL_CLASS = None
            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False
            _AUTO_MODEL_CLASS = None
            print("Warning: transformers not available. Advanced document processing features will be disabled.")

import numpy as np
from ..models.base import BaseLLM
from .document_chunkers import *
from .document_embeddings import *

try:
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False


@dataclass
class ProcessingConfig:
    """Configuration for document processing operations."""
    
    # Chunking configuration
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    chunk_overlap: int = 50
    similarity_threshold: float = 0.7
    
    # Metadata extraction
    extract_metadata: bool = True
    extract_entities: bool = True
    extract_key_phrases: bool = True
    extract_statistics: bool = True
    
    # Embedding configuration
    generate_embeddings: bool = True
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None
    
    # Processing options
    remove_html: bool = True
    remove_urls: bool = False
    remove_emails: bool = False
    remove_phone_numbers: bool = False
    normalize_whitespace: bool = True
    lowercase: bool = False
    
    # Language processing
    language: str = "en"
    use_spacy: bool = True
    spacy_model: str = "en_core_web_sm"
    
    # Advanced options
    merge_similar_chunks: bool = True
    max_merged_tokens: Optional[int] = None
    preserve_formatting: bool = False
    include_original_text: bool = False
    
    # Performance settings
    batch_size: int = 10
    max_workers: Optional[int] = None
    timeout: Optional[float] = None
    
    # Custom processing functions
    preprocess_fn: Optional[Callable[[str], str]] = None
    postprocess_fn: Optional[Callable[[List[DocumentChunk]], List[DocumentChunk]]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.min_chunk_size > self.max_chunk_size:
            raise ValueError("min_chunk_size cannot be greater than max_chunk_size")
        
        if self.chunk_overlap >= self.max_chunk_size:
            raise ValueError("chunk_overlap must be less than max_chunk_size")
        
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'chunking_strategy': self.chunking_strategy.value,
            'min_chunk_size': self.min_chunk_size,
            'max_chunk_size': self.max_chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'similarity_threshold': self.similarity_threshold,
            'extract_metadata': self.extract_metadata,
            'extract_entities': self.extract_entities,
            'extract_key_phrases': self.extract_key_phrases,
            'extract_statistics': self.extract_statistics,
            'generate_embeddings': self.generate_embeddings,
            'embedding_model': self.embedding_model,
            'embedding_dimension': self.embedding_dimension,
            'remove_html': self.remove_html,
            'remove_urls': self.remove_urls,
            'remove_emails': self.remove_emails,
            'remove_phone_numbers': self.remove_phone_numbers,
            'normalize_whitespace': self.normalize_whitespace,
            'lowercase': self.lowercase,
            'language': self.language,
            'use_spacy': self.use_spacy,
            'spacy_model': self.spacy_model,
            'merge_similar_chunks': self.merge_similar_chunks,
            'max_merged_tokens': self.max_merged_tokens,
            'preserve_formatting': self.preserve_formatting,
            'include_original_text': self.include_original_text,
            'batch_size': self.batch_size,
            'max_workers': self.max_workers,
            'timeout': self.timeout
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProcessingConfig':
        """Create configuration from dictionary."""
        # Convert string strategy back to enum
        if 'chunking_strategy' in config_dict and isinstance(config_dict['chunking_strategy'], str):
            config_dict['chunking_strategy'] = ChunkingStrategy(config_dict['chunking_strategy'])
        
        return cls(**config_dict)
    
    def get_chunker_config(self) -> Dict[str, Any]:
        """Get configuration specific to chunking operations."""
        return {
            'min_chunk_size': self.min_chunk_size,
            'max_chunk_size': self.max_chunk_size,
            'similarity_threshold': self.similarity_threshold,
            'chunk_overlap': self.chunk_overlap
        }
    
    def get_metadata_config(self) -> Dict[str, Any]:
        """Get configuration specific to metadata extraction."""
        return {
            'extract_entities': self.extract_entities,
            'extract_key_phrases': self.extract_key_phrases,
            'extract_statistics': self.extract_statistics,
            'language': self.language,
            'use_spacy': self.use_spacy,
            'spacy_model': self.spacy_model
        }


class EnhancedDocumentProcessor:
    """Enhanced document processing with multiple strategies."""

    def __init__(
        self,
        model: BaseLLM,
        config: Optional[ProcessingConfig] = None,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        metadata_extractor: Optional[MetadataExtractor] = None,
        **kwargs
    ):
        self.model = model
        self.config = config or ProcessingConfig()
        self.chunking_strategy = chunking_strategy
        self.metadata_extractor = metadata_extractor or MetadataExtractor()
        
        # Use config for chunker initialization
        chunker_config = self.config.get_chunker_config()
        self.semantic_chunker = SemanticChunker(model, **chunker_config, **kwargs)
        self.kwargs = kwargs

    async def process_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[DocumentChunk]:
        """
        Process document with enhanced chunking and metadata extraction.
        
        Args:
            text: Input document text
            metadata: Optional initial metadata
            **kwargs: Additional processing parameters
            
        Returns:
            List of processed document chunks
        """
        # Preprocess text if configured
        if self.config.preprocess_fn:
            text = self.config.preprocess_fn(text)
        
        # Extract metadata if configured
        if self.config.extract_metadata:
            extracted_metadata = self.metadata_extractor.extract_metadata(text)
            if metadata:
                extracted_metadata.update(metadata)
        else:
            extracted_metadata = metadata or {}
        
        # Chunk document based on strategy
        if self.chunking_strategy == ChunkingStrategy.SEMANTIC:
            chunks = await self.semantic_chunker.chunk_document(
                text,
                metadata=extracted_metadata,
                **kwargs
            )
        else:
            # Implement other chunking strategies
            raise NotImplementedError(
                f"Chunking strategy {self.chunking_strategy} not implemented"
            )
        
        # Generate embeddings for chunks if configured
        if self.config.generate_embeddings:
            for chunk in chunks:
                chunk.embedding = await self.model.embeddings([chunk.text])[0]
        
        # Postprocess chunks if configured
        if self.config.postprocess_fn:
            chunks = self.config.postprocess_fn(chunks)
        
        return chunks

    async def process_documents(
        self,
        documents: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[List[DocumentChunk]]:
        """
        Process multiple documents in parallel.
        
        Args:
            documents: List of document texts
            metadata_list: Optional list of metadata dictionaries
            **kwargs: Additional processing parameters
            
        Returns:
            List of processed document chunks for each document
        """
        if metadata_list is None:
            metadata_list = [{}] * len(documents)
        
        # Process documents in parallel
        tasks = [
            self.process_document(doc, meta, **kwargs)
            for doc, meta in zip(documents, metadata_list)
        ]
        
        return await asyncio.gather(*tasks)

    async def merge_chunks(
        self,
        chunks: List[DocumentChunk],
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[DocumentChunk]:
        """
        Merge chunks based on semantic similarity and token budget.
        
        Args:
            chunks: List of document chunks
            max_tokens: Optional maximum tokens per merged chunk
            **kwargs: Additional merging parameters
            
        Returns:
            List of merged chunks
        """
        if not chunks or not self.config.merge_similar_chunks:
            return chunks
        
        # Use config max_tokens if not provided
        if max_tokens is None:
            max_tokens = self.config.max_merged_tokens
        
        # Sort chunks by semantic score
        sorted_chunks = sorted(chunks, key=lambda x: x.semantic_score or 0, reverse=True)
        
        merged_chunks = []
        current_chunk = sorted_chunks[0]
        
        for next_chunk in sorted_chunks[1:]:
            # Check if chunks should be merged
            if self._should_merge_chunks(current_chunk, next_chunk, max_tokens):
                current_chunk = self._merge_two_chunks(current_chunk, next_chunk)
            else:
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        merged_chunks.append(current_chunk)
        return merged_chunks

    def _should_merge_chunks(
        self,
        chunk1: DocumentChunk,
        chunk2: DocumentChunk,
        max_tokens: Optional[int]
    ) -> bool:
        """Determine if two chunks should be merged."""
        if not chunk1.embedding or not chunk2.embedding:
            return False
        
        # Check semantic similarity
        similarity = self.semantic_chunker._cosine_similarity(
            chunk1.embedding,
            chunk2.embedding
        )
        
        # Check token count if max_tokens is specified
        if max_tokens:
            combined_tokens = len(
                self.semantic_chunker.tokenizer.encode(
                    chunk1.text + " " + chunk2.text
                )
            )
            if combined_tokens > max_tokens:
                return False
        
        return similarity >= self.config.similarity_threshold

    def _merge_two_chunks(self, chunk1: DocumentChunk, chunk2: DocumentChunk) -> DocumentChunk:
        """Merge two chunks into one."""
        return DocumentChunk(
            text=chunk1.text + " " + chunk2.text,
            metadata={**chunk1.metadata, **chunk2.metadata},
            chunk_id=f"merged_{chunk1.chunk_id}_{chunk2.chunk_id}",
            parent_id=None,
            semantic_score=min(chunk1.semantic_score or 0, chunk2.semantic_score or 0),
            embedding=np.mean([chunk1.embedding, chunk2.embedding], axis=0)
            if chunk1.embedding and chunk2.embedding
            else None
        )

# Backward compatibility alias
DocumentProcessor = EnhancedDocumentProcessor

