"""
Advanced memory system for RAG with token-aware pruning and cross-session support.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import time
from datetime import datetime
import numpy as np
from transformers import AutoTokenizer
from ..models.base import BaseLLM

@dataclass
class MemoryItem:
    """Represents a memory item with metadata."""
    content: str
    timestamp: float
    importance: float
    tokens: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class ConversationTurn:
    """Represents a conversation turn with context."""
    query: str
    response: str
    context: List[Dict[str, Any]]
    timestamp: float
    metadata: Dict[str, Any]

class MemoryType(Enum):
    """Types of memory storage."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"

@runtime_checkable
class MemoryStore(Protocol):
    """Protocol for memory storage backends."""
    async def add(self, item: MemoryItem) -> None:
        """Add item to memory store."""
        ...
    
    async def get(self, query: str, k: int = 5) -> List[MemoryItem]:
        """Retrieve relevant items from memory store."""
        ...
    
    async def clear(self) -> None:
        """Clear memory store."""
        ...

class TokenAwareMemory:
    """Memory system with token-aware pruning."""

    def __init__(
        self,
        model: BaseLLM,
        max_tokens: int = 4000,
        importance_threshold: float = 0.5,
        **kwargs
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.importance_threshold = importance_threshold
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.short_term: List[MemoryItem] = []
        self.long_term: List[MemoryItem] = []
        self.working: List[MemoryItem] = []
        self.conversation_history: List[ConversationTurn] = []
        self.kwargs = kwargs

    async def add_to_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Add content to memory with automatic pruning.
        
        Args:
            content: Content to remember
            memory_type: Type of memory to use
            metadata: Optional metadata
            **kwargs: Additional parameters
        """
        # Calculate tokens and importance
        tokens = len(self.tokenizer.encode(content))
        importance = await self._calculate_importance(content, **kwargs)
        
        # Create memory item
        item = MemoryItem(
            content=content,
            timestamp=time.time(),
            importance=importance,
            tokens=tokens,
            metadata=metadata or {},
            embedding=await self.model.embeddings([content])[0]
        )
        
        # Add to appropriate memory
        if memory_type == MemoryType.SHORT_TERM:
            self.short_term.append(item)
        elif memory_type == MemoryType.LONG_TERM:
            self.long_term.append(item)
        else:
            self.working.append(item)
        
        # Prune if necessary
        await self._prune_memory(memory_type)

    async def get_relevant_memory(
        self,
        query: str,
        k: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
        **kwargs
    ) -> List[MemoryItem]:
        """
        Retrieve relevant memory items.
        
        Args:
            query: Query to find relevant memories
            k: Number of items to retrieve
            memory_types: Optional list of memory types to search
            **kwargs: Additional parameters
            
        Returns:
            List of relevant memory items
        """
        if memory_types is None:
            memory_types = [MemoryType.WORKING, MemoryType.SHORT_TERM, MemoryType.LONG_TERM]
        
        # Get items from specified memory types
        all_items = []
        for memory_type in memory_types:
            if memory_type == MemoryType.SHORT_TERM:
                all_items.extend(self.short_term)
            elif memory_type == MemoryType.LONG_TERM:
                all_items.extend(self.long_term)
            else:
                all_items.extend(self.working)
        
        if not all_items:
            return []
        
        # Calculate relevance scores
        query_embedding = await self.model.embeddings([query])[0]
        scores = [
            self._cosine_similarity(query_embedding, item.embedding)
            for item in all_items
        ]
        
        # Combine with importance scores
        combined_scores = [
            score * item.importance
            for score, item in zip(scores, all_items)
        ]
        
        # Get top k items
        top_k_indices = np.argsort(combined_scores)[-k:][::-1]
        return [all_items[i] for i in top_k_indices]

    async def add_conversation_turn(
        self,
        query: str,
        response: str,
        context: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Add a conversation turn to history.
        
        Args:
            query: User query
            response: System response
            context: Retrieved context
            metadata: Optional metadata
            **kwargs: Additional parameters
        """
        turn = ConversationTurn(
            query=query,
            response=response,
            context=context,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        self.conversation_history.append(turn)
        
        # Add to memory if important
        importance = await self._calculate_importance(
            f"Q: {query}\nA: {response}",
            **kwargs
        )
        if importance > self.importance_threshold:
            await self.add_to_memory(
                f"Q: {query}\nA: {response}",
                memory_type=MemoryType.LONG_TERM,
                metadata={
                    "type": "conversation",
                    "importance": importance,
                    **(metadata or {})
                }
            )

    async def get_conversation_history(
        self,
        max_turns: Optional[int] = None,
        **kwargs
    ) -> List[ConversationTurn]:
        """
        Get conversation history.
        
        Args:
            max_turns: Optional maximum number of turns
            **kwargs: Additional parameters
            
        Returns:
            List of conversation turns
        """
        if max_turns is None:
            return self.conversation_history
        return self.conversation_history[-max_turns:]

    async def _prune_memory(self, memory_type: MemoryType) -> None:
        """Prune memory based on token budget and importance."""
        if memory_type == MemoryType.SHORT_TERM:
            memory = self.short_term
        elif memory_type == MemoryType.LONG_TERM:
            memory = self.long_term
        else:
            memory = self.working
        
        # Calculate total tokens
        total_tokens = sum(item.tokens for item in memory)
        
        if total_tokens > self.max_tokens:
            # Sort by importance and timestamp
            memory.sort(key=lambda x: (x.importance, x.timestamp))
            
            # Remove items until under token budget
            while total_tokens > self.max_tokens and memory:
                item = memory.pop(0)
                total_tokens -= item.tokens

    async def _calculate_importance(
        self,
        content: str,
        **kwargs
    ) -> float:
        """
        Calculate importance score for content.
        
        Args:
            content: Content to score
            **kwargs: Additional parameters
            
        Returns:
            Importance score between 0 and 1
        """
        # This is a placeholder implementation
        # In practice, you might want to use:
        # 1. LLM-based importance scoring
        # 2. Heuristics based on content type
        # 3. User feedback
        # 4. Query relevance
        return 0.5

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class PersistentMemoryStore:
    """Persistent memory storage with cross-session support."""

    def __init__(
        self,
        storage_path: str,
        model: BaseLLM,
        **kwargs
    ):
        self.storage_path = storage_path
        self.model = model
        self.memory = TokenAwareMemory(model, **kwargs)
        self.kwargs = kwargs

    async def save(self) -> None:
        """Save memory state to disk."""
        state = {
            "short_term": [
                {
                    "content": item.content,
                    "timestamp": item.timestamp,
                    "importance": item.importance,
                    "tokens": item.tokens,
                    "metadata": item.metadata
                }
                for item in self.memory.short_term
            ],
            "long_term": [
                {
                    "content": item.content,
                    "timestamp": item.timestamp,
                    "importance": item.importance,
                    "tokens": item.tokens,
                    "metadata": item.metadata
                }
                for item in self.memory.long_term
            ],
            "conversation_history": [
                {
                    "query": turn.query,
                    "response": turn.response,
                    "context": turn.context,
                    "timestamp": turn.timestamp,
                    "metadata": turn.metadata
                }
                for turn in self.memory.conversation_history
            ]
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(state, f)

    async def load(self) -> None:
        """Load memory state from disk."""
        try:
            with open(self.storage_path, 'r') as f:
                state = json.load(f)
            
            # Reconstruct memory items
            for item in state["short_term"]:
                await self.memory.add_to_memory(
                    content=item["content"],
                    memory_type=MemoryType.SHORT_TERM,
                    metadata=item["metadata"]
                )
            
            for item in state["long_term"]:
                await self.memory.add_to_memory(
                    content=item["content"],
                    memory_type=MemoryType.LONG_TERM,
                    metadata=item["metadata"]
                )
            
            # Reconstruct conversation history
            for turn in state["conversation_history"]:
                await self.memory.add_conversation_turn(
                    query=turn["query"],
                    response=turn["response"],
                    context=turn["context"],
                    metadata=turn["metadata"]
                )
        
        except FileNotFoundError:
            # Initialize new memory if no saved state
            pass

    async def __aenter__(self):
        """Context manager entry."""
        await self.load()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.save() 