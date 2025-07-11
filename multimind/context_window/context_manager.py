"""
Advanced context manager module for RAG systems.
Handles both context window management and vector database operations.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import numpy as np
from datetime import datetime
import faiss
import hnswlib
import tiktoken
import torch
from transformers import AutoTokenizer, AutoModel
import logging
from pathlib import Path
import pickle
from ..models.base import BaseLLM
from ..embeddings.embedding import EmbeddingModel, EmbeddingConfig
from ..vector_store import VectorStore, VectorStoreConfig

# Try to import Redis and Redis search modules, but handle gracefully if not available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    if REDIS_AVAILABLE:
        from redis.commands.search.field import VectorField, TagField, TextField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        REDIS_SEARCH_AVAILABLE = True
    else:
        REDIS_SEARCH_AVAILABLE = False
        VectorField = TagField = TextField = IndexDefinition = IndexType = None
except ImportError:
    REDIS_SEARCH_AVAILABLE = False
    VectorField = TagField = TextField = IndexDefinition = IndexType = None

@dataclass
class ContextConfig:
    """General configuration for context management."""
    max_tokens: int = 2048
    chunk_size: int = 256
    overlap_tokens: int = 32
    compression_ratio: float = 0.5
    relevance_threshold: float = 0.7
    memory_limit: int = 10000
    custom_params: Dict[str, Any] = None

@dataclass
class ContextWindowConfig:
    """Configuration for context window management."""
    max_tokens: int
    overlap_tokens: int
    chunk_size: int
    chunk_overlap: int
    compression_ratio: float
    relevance_threshold: float
    memory_limit: int
    custom_params: Dict[str, Any]

@dataclass
class ContextChunk:
    """Represents a chunk of context."""
    content: str
    metadata: Dict[str, Any]
    tokens: int
    embedding: Optional[List[float]]
    relevance_score: float
    timestamp: float

@dataclass
class ContextWindow:
    """Represents a context window."""
    chunks: List[ContextChunk]
    metadata: Dict[str, Any]
    total_tokens: int
    last_updated: float

class CompressionStrategy(Enum):
    """Types of context compression strategies."""
    SEMANTIC = "semantic"
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    HYBRID = "hybrid"

class ContextManager:
    """Advanced context manager with window management and vector database operations."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm: Optional[BaseLLM] = None,
        vector_store: Optional[VectorStore] = None,
        config: Optional[ContextWindowConfig] = None,
        **kwargs
    ):
        """
        Initialize context manager.
        
        Args:
            embedding_model: Embedding model for vector operations
            llm: Optional LLM for advanced features
            vector_store: Optional vector store for persistence
            config: Optional context window configuration
            **kwargs: Additional parameters
        """
        self.embedding_model = embedding_model
        self.llm = llm
        self.vector_store = vector_store
        self.config = config or self._get_default_config()
        self.kwargs = kwargs
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize context window
        self.window = ContextWindow(
            chunks=[],
            metadata={},
            total_tokens=0,
            last_updated=datetime.now().timestamp()
        )
        
        # Initialize cache
        self.cache = {
            "embeddings": {},
            "relevance_scores": {},
            "compressed_chunks": {}
        }

    def _get_default_config(self) -> ContextWindowConfig:
        """Get default context window configuration."""
        return ContextWindowConfig(
            max_tokens=4000,
            overlap_tokens=200,
            chunk_size=1000,
            chunk_overlap=100,
            compression_ratio=0.5,
            relevance_threshold=0.7,
            memory_limit=10000,
            custom_params={}
        )

    async def add_to_context(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Add content to context window.
        
        Args:
            content: Content to add
            metadata: Optional metadata
            **kwargs: Additional parameters
        """
        # Create chunks
        chunks = await self._create_chunks(
            content,
            metadata or {}
        )
        
        # Add chunks to window
        for chunk in chunks:
            await self._add_chunk(chunk)
        
        # Update window metadata
        self.window.last_updated = datetime.now().timestamp()
        self.window.metadata.update(metadata or {})

    async def _create_chunks(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[ContextChunk]:
        """Create chunks from content."""
        # Tokenize content
        tokens = self.tokenizer.encode(content)
        
        # Create chunks
        chunks = []
        for i in range(0, len(tokens), self.config.chunk_size - self.config.chunk_overlap):
            # Get chunk tokens
            chunk_tokens = tokens[i:i + self.config.chunk_size]
            
            # Decode chunk
            chunk_content = self.tokenizer.decode(chunk_tokens)
            
            # Create chunk
            chunk = ContextChunk(
                content=chunk_content,
                metadata={
                    **metadata,
                    "chunk_index": len(chunks),
                    "token_count": len(chunk_tokens)
                },
                tokens=len(chunk_tokens),
                embedding=None,
                relevance_score=1.0,
                timestamp=datetime.now().timestamp()
            )
            
            chunks.append(chunk)
        
        return chunks

    async def _add_chunk(self, chunk: ContextChunk) -> None:
        """Add chunk to context window."""
        # Generate embedding if needed
        if chunk.embedding is None:
            chunk.embedding = await self.embedding_model.generate_embedding(
                chunk.content,
                EmbeddingConfig(
                    model_name=self.embedding_model.model_name,
                    model_type=self.embedding_model.model_type.value,
                    batch_size=1,
                    max_length=512,
                    normalize=True,
                    device="cuda" if self.embedding_model.device == "cuda" else "cpu",
                    cache_dir=None,
                    custom_params={}
                )
            )
        
        # Add to window
        self.window.chunks.append(chunk)
        self.window.total_tokens += chunk.tokens
        
        # Check window size
        if self.window.total_tokens > self.config.max_tokens:
            await self._prune_window()
        
        # Add to vector store if available
        if self.vector_store:
            await self.vector_store.add_vectors(
                vectors=[chunk.embedding],
                metadatas=[chunk.metadata],
                documents=[{"content": chunk.content, "metadata": chunk.metadata}],
                ids=[f"chunk_{len(self.window.chunks)}"]
            )

    async def _prune_window(self) -> None:
        """Prune context window to maintain size limits."""
        if not self.window.chunks:
            return
        
        # Calculate relevance scores if needed
        if not all(chunk.relevance_score < 1.0 for chunk in self.window.chunks):
            await self._update_relevance_scores()
        
        # Sort chunks by relevance
        self.window.chunks.sort(
            key=lambda x: x.relevance_score,
            reverse=True
        )
        
        # Remove chunks until window size is acceptable
        while self.window.total_tokens > self.config.max_tokens:
            if not self.window.chunks:
                break
            
            # Remove least relevant chunk
            removed = self.window.chunks.pop()
            self.window.total_tokens -= removed.tokens
            
            # Remove from vector store if available
            if self.vector_store:
                await self.vector_store.delete_vectors(
                    [f"chunk_{len(self.window.chunks) + 1}"]
                )

    async def _update_relevance_scores(self) -> None:
        """Update relevance scores for chunks."""
        if not self.llm:
            return
        
        # Generate relevance scores using LLM
        for chunk in self.window.chunks:
            prompt = f"""
            Score the relevance of the following content to the current context.
            Consider:
            1. Information recency
            2. Semantic importance
            3. Contextual relevance
            
            Content:
            {chunk.content}
            
            Relevance score (0-1):
            """
            
            score_text = await self.llm.generate(prompt)
            try:
                score = float(score_text.strip())
                chunk.relevance_score = max(0.0, min(1.0, score))
            except ValueError:
                chunk.relevance_score = 0.5

    async def compress_context(
        self,
        strategy: str = CompressionStrategy.SEMANTIC.value,
        **kwargs
    ) -> None:
        """
        Compress context window.
        
        Args:
            strategy: Compression strategy
            **kwargs: Additional parameters
        """
        if not self.window.chunks:
            return
        
        if strategy == CompressionStrategy.SEMANTIC.value:
            await self._semantic_compression()
        
        elif strategy == CompressionStrategy.EXTRACTIVE.value:
            await self._extractive_compression()
        
        elif strategy == CompressionStrategy.ABSTRACTIVE.value:
            await self._abstractive_compression()
        
        elif strategy == CompressionStrategy.HYBRID.value:
            await self._hybrid_compression()
        
        else:
            raise ValueError(f"Unsupported compression strategy: {strategy}")

    async def _semantic_compression(self) -> None:
        """Compress context using semantic similarity."""
        if not self.llm:
            return
        
        # Group similar chunks
        groups = []
        current_group = []
        
        for chunk in self.window.chunks:
            if not current_group:
                current_group.append(chunk)
            else:
                # Check similarity with group
                similarity = await self._calculate_similarity(
                    chunk,
                    current_group[0]
                )
                
                if similarity > self.config.relevance_threshold:
                    current_group.append(chunk)
                else:
                    groups.append(current_group)
                    current_group = [chunk]
        
        if current_group:
            groups.append(current_group)
        
        # Compress each group
        compressed_chunks = []
        for group in groups:
            if len(group) == 1:
                compressed_chunks.append(group[0])
            else:
                # Combine similar chunks
                combined = await self._combine_chunks(group)
                compressed_chunks.append(combined)
        
        # Update window
        self.window.chunks = compressed_chunks
        self.window.total_tokens = sum(
            chunk.tokens for chunk in compressed_chunks
        )

    async def _extractive_compression(self) -> None:
        """Compress context using extractive summarization."""
        if not self.llm:
            return
        
        # Generate summary for each chunk
        for chunk in self.window.chunks:
            prompt = f"""
            Summarize the following content, preserving key information while reducing length.
            Target length: {int(chunk.tokens * self.config.compression_ratio)} tokens.
            
            Content:
            {chunk.content}
            
            Summary:
            """
            
            summary = await self.llm.generate(prompt)
            
            # Update chunk
            chunk.content = summary.text
            chunk.tokens = len(self.tokenizer.encode(summary.text))
            
            # Update embedding
            chunk.embedding = await self.embedding_model.generate_embedding(
                chunk.content,
                EmbeddingConfig(
                    model_name=self.embedding_model.model_name,
                    model_type=self.embedding_model.model_type.value,
                    batch_size=1,
                    max_length=512,
                    normalize=True,
                    device="cuda" if self.embedding_model.device == "cuda" else "cpu",
                    cache_dir=None,
                    custom_params={}
                )
            )

    async def _abstractive_compression(self) -> None:
        """Compress context using abstractive summarization."""
        if not self.llm:
            return
        
        # Generate abstractive summary
        prompt = f"""
        Generate a concise summary of the following content, focusing on key insights and main points.
        Target length: {int(self.window.total_tokens * self.config.compression_ratio)} tokens.
        
        Content:
        {self._format_context()}
        
        Summary:
        """
        
        summary = await self.llm.generate(prompt)
        
        # Create new chunk
        new_chunk = ContextChunk(
            content=summary.text,
            metadata={
                "compression_type": "abstractive",
                "original_tokens": self.window.total_tokens,
                "compressed_tokens": len(self.tokenizer.encode(summary.text))
            },
            tokens=len(self.tokenizer.encode(summary.text)),
            embedding=None,
            relevance_score=1.0,
            timestamp=datetime.now().timestamp()
        )
        
        # Update window
        self.window.chunks = [new_chunk]
        self.window.total_tokens = new_chunk.tokens

    async def _hybrid_compression(self) -> None:
        """Compress context using hybrid approach."""
        # First apply semantic compression
        await self._semantic_compression()
        
        # Then apply abstractive compression
        await self._abstractive_compression()

    async def _calculate_similarity(
        self,
        chunk1: ContextChunk,
        chunk2: ContextChunk
    ) -> float:
        """Calculate similarity between chunks."""
        if chunk1.embedding is None or chunk2.embedding is None:
            return 0.0
        
        # Calculate cosine similarity
        vec1 = np.array(chunk1.embedding)
        vec2 = np.array(chunk2.embedding)
        
        similarity = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )
        
        return float(similarity)

    async def _combine_chunks(
        self,
        chunks: List[ContextChunk]
    ) -> ContextChunk:
        """Combine similar chunks."""
        if not self.llm:
            # Simple concatenation if no LLM available
            combined_content = "\n".join(
                chunk.content for chunk in chunks
            )
            combined_tokens = sum(
                chunk.tokens for chunk in chunks
            )
        else:
            # Use LLM to combine chunks
            prompt = f"""
            Combine the following related content chunks into a single coherent piece.
            Preserve all important information while eliminating redundancy.
            
            Chunks:
            {self._format_chunks(chunks)}
            
            Combined content:
            """
            
            combined = await self.llm.generate(prompt)
            combined_content = combined.text
            combined_tokens = len(self.tokenizer.encode(combined_content))
        
        # Create combined chunk
        return ContextChunk(
            content=combined_content,
            metadata={
                "compression_type": "semantic",
                "original_chunks": len(chunks),
                "original_tokens": sum(
                    chunk.tokens for chunk in chunks
                )
            },
            tokens=combined_tokens,
            embedding=None,
            relevance_score=max(
                chunk.relevance_score for chunk in chunks
            ),
            timestamp=datetime.now().timestamp()
        )

    def _format_context(self) -> str:
        """Format context for LLM input."""
        return "\n\n".join(
            f"Chunk {i+1}:\n{chunk.content}"
            for i, chunk in enumerate(self.window.chunks)
        )

    def _format_chunks(self, chunks: List[ContextChunk]) -> str:
        """Format chunks for LLM input."""
        return "\n\n".join(
            f"Chunk {i+1}:\n{chunk.content}"
            for i, chunk in enumerate(chunks)
        )

    async def search_context(
        self,
        query: str,
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[ContextChunk]:
        """
        Search context window.
        
        Args:
            query: Search query
            k: Number of results
            filter_criteria: Optional filtering criteria
            **kwargs: Additional parameters
            
        Returns:
            List of relevant chunks
        """
        if not self.window.chunks:
            return []
        
        if self.vector_store:
            # Use vector store for search
            query_embedding = await self.embedding_model.generate_embedding(
                query,
                EmbeddingConfig(
                    model_name=self.embedding_model.model_name,
                    model_type=self.embedding_model.model_type.value,
                    batch_size=1,
                    max_length=512,
                    normalize=True,
                    device="cuda" if self.embedding_model.device == "cuda" else "cpu",
                    cache_dir=None,
                    custom_params={}
                )
            )
            
            results = await self.vector_store.search(
                query_embedding,
                k=k,
                filter_criteria=filter_criteria
            )
            
            # Convert results to chunks
            chunks = []
            for result in results:
                chunk = ContextChunk(
                    content=result.document["content"],
                    metadata=result.metadata,
                    tokens=len(self.tokenizer.encode(result.document["content"])),
                    embedding=result.vector,
                    relevance_score=result.score,
                    timestamp=datetime.now().timestamp()
                )
                chunks.append(chunk)
            
            return chunks
        
        else:
            # Use local search
            query_embedding = await self.embedding_model.generate_embedding(
                query,
                EmbeddingConfig(
                    model_name=self.embedding_model.model_name,
                    model_type=self.embedding_model.model_type.value,
                    batch_size=1,
                    max_length=512,
                    normalize=True,
                    device="cuda" if self.embedding_model.device == "cuda" else "cpu",
                    cache_dir=None,
                    custom_params={}
                )
            )
            
            # Calculate similarities
            similarities = []
            for chunk in self.window.chunks:
                if chunk.embedding is None:
                    chunk.embedding = await self.embedding_model.generate_embedding(
                        chunk.content,
                        EmbeddingConfig(
                            model_name=self.embedding_model.model_name,
                            model_type=self.embedding_model.model_type.value,
                            batch_size=1,
                            max_length=512,
                            normalize=True,
                            device="cuda" if self.embedding_model.device == "cuda" else "cpu",
                            cache_dir=None,
                            custom_params={}
                        )
                    )
                
                similarity = await self._calculate_similarity(
                    ContextChunk(
                        content="",
                        metadata={},
                        tokens=0,
                        embedding=query_embedding,
                        relevance_score=0.0,
                        timestamp=0.0
                    ),
                    chunk
                )
                
                similarities.append((chunk, similarity))
            
            # Sort by similarity
            similarities.sort(
                key=lambda x: x[1],
                reverse=True
            )
            
            return [chunk for chunk, _ in similarities[:k]]

    async def clear_context(self) -> None:
        """Clear context window."""
        self.window = ContextWindow(
            chunks=[],
            metadata={},
            total_tokens=0,
            last_updated=datetime.now().timestamp()
        )
        
        # Clear vector store if available
        if self.vector_store:
            # This is a placeholder - implement proper clearing based on your vector store
            pass

    async def persist_context(self, path: str) -> None:
        """
        Persist context to disk.
        
        Args:
            path: Path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save window
        with open(path / "window.pkl", "wb") as f:
            pickle.dump(self.window, f)
        
        # Save cache
        with open(path / "cache.pkl", "wb") as f:
            pickle.dump(self.cache, f)
        
        # Save vector store if available
        if self.vector_store:
            await self.vector_store.persist(str(path / "vector_store"))

    @classmethod
    async def load_context(
        cls,
        path: str,
        embedding_model: EmbeddingModel,
        llm: Optional[BaseLLM] = None,
        vector_store: Optional[VectorStore] = None,
        config: Optional[ContextWindowConfig] = None
    ) -> "ContextManager":
        """
        Load context from disk.
        
        Args:
            path: Path to load from
            embedding_model: Embedding model
            llm: Optional LLM
            vector_store: Optional vector store
            config: Optional configuration
            
        Returns:
            Loaded context manager
        """
        path = Path(path)
        
        # Create instance
        manager = cls(
            embedding_model=embedding_model,
            llm=llm,
            vector_store=vector_store,
            config=config
        )
        
        # Load window
        with open(path / "window.pkl", "rb") as f:
            manager.window = pickle.load(f)
        
        # Load cache
        with open(path / "cache.pkl", "rb") as f:
            manager.cache = pickle.load(f)
        
        # Load vector store if available
        if vector_store and (path / "vector_store").exists():
            manager.vector_store = await VectorStore.load(
                str(path / "vector_store"),
                vector_store.config
            )
        
        return manager