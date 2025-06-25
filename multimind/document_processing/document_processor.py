"""
Enhanced document processing with semantic chunking and metadata extraction.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import re
from dataclasses import dataclass
from enum import Enum
import spacy
from bs4 import BeautifulSoup
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
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



class EnhancedDocumentProcessor:
    """Enhanced document processing with multiple strategies."""

    def __init__(
        self,
        model: BaseLLM,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        metadata_extractor: Optional[MetadataExtractor] = None,
        **kwargs
    ):
        self.model = model
        self.chunking_strategy = chunking_strategy
        self.metadata_extractor = metadata_extractor or MetadataExtractor()
        self.semantic_chunker = SemanticChunker(model, **kwargs)
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
        # Extract metadata
        extracted_metadata = self.metadata_extractor.extract_metadata(text)
        if metadata:
            extracted_metadata.update(metadata)
        
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
        
        # Generate embeddings for chunks
        for chunk in chunks:
            chunk.embedding = await self.model.embeddings([chunk.text])[0]
        
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
        if not chunks:
            return []
        
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
        
        return similarity >= self.semantic_chunker.similarity_threshold

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

