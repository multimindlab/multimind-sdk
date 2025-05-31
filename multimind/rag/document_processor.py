"""
Enhanced document processing with semantic chunking and metadata extraction.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import re
from dataclasses import dataclass
from enum import Enum
import spacy
from bs4 import BeautifulSoup
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
import numpy as np
from ..models.base import BaseLLM

@dataclass
class DocumentChunk:
    """Represents a processed document chunk."""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    parent_id: Optional[str]
    semantic_score: Optional[float] = None
    embedding: Optional[List[float]] = None

class ChunkingStrategy(Enum):
    """Different document chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    SLIDING_WINDOW = "sliding_window"

class MetadataExtractor:
    """Extracts and enriches document metadata."""

    def __init__(self, nlp_model: Optional[str] = "en_core_web_sm"):
        self.nlp = spacy.load(nlp_model) if nlp_model else None

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from text using NLP.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted metadata
        """
        if not self.nlp:
            return {}

        doc = self.nlp(text)
        
        # Extract entities
        entities = {
            ent.label_: [e.text for e in doc.ents if e.label_ == ent.label_]
            for ent in doc.ents
        }
        
        # Extract key phrases (noun chunks)
        key_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Extract document statistics
        stats = {
            "word_count": len(doc),
            "sentence_count": len(list(doc.sents)),
            "avg_word_length": np.mean([len(token.text) for token in doc]),
            "unique_words": len(set(token.text.lower() for token in doc))
        }
        
        return {
            "entities": entities,
            "key_phrases": key_phrases,
            "statistics": stats
        }

class SemanticChunker:
    """Implements semantic document chunking."""

    def __init__(
        self,
        model: BaseLLM,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        similarity_threshold: float = 0.7,
        **kwargs
    ):
        self.model = model
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.summarizer = AutoModelForSeq2SeqGeneration.from_pretrained("facebook/bart-large-cnn")

    async def chunk_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[DocumentChunk]:
        """
        Chunk document semantically based on content similarity.
        
        Args:
            text: Input document text
            metadata: Optional document metadata
            **kwargs: Additional chunking parameters
            
        Returns:
            List of semantic document chunks
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        # Generate embeddings for sentences
        sentence_embeddings = await self.model.embeddings(sentences)
        
        # Group similar sentences
        chunks = self._group_similar_sentences(sentences, sentence_embeddings)
        
        # Create DocumentChunk objects
        return [
            DocumentChunk(
                text=chunk_text,
                metadata=metadata or {},
                chunk_id=f"chunk_{i}",
                parent_id=None,
                semantic_score=self._calculate_semantic_score(chunk_text)
            )
            for i, chunk_text in enumerate(chunks)
        ]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLP."""
        doc = spacy.load("en_core_web_sm")(text)
        return [sent.text.strip() for sent in doc.sents]

    def _group_similar_sentences(
        self,
        sentences: List[str],
        embeddings: List[List[float]]
    ) -> List[str]:
        """Group similar sentences into chunks."""
        chunks = []
        current_chunk = []
        current_embedding = None
        
        for sentence, embedding in zip(sentences, embeddings):
            if not current_chunk:
                current_chunk.append(sentence)
                current_embedding = embedding
            else:
                # Calculate similarity with current chunk
                similarity = self._cosine_similarity(current_embedding, embedding)
                
                if similarity >= self.similarity_threshold:
                    current_chunk.append(sentence)
                    # Update chunk embedding
                    current_embedding = np.mean([current_embedding, embedding], axis=0)
                else:
                    # Start new chunk
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_embedding = embedding
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _calculate_semantic_score(self, text: str) -> float:
        """Calculate semantic coherence score for a chunk."""
        # This is a placeholder implementation
        # In practice, you might want to use more sophisticated methods
        return 1.0

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