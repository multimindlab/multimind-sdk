"""
Adaptive Context Windows Memory implementation.
"""

from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
from .base import BaseMemory
from .vector_store import VectorStoreMemory

class AdaptiveMemory(BaseMemory):
    """Memory implementation with adaptive context windows."""

    def __init__(
        self,
        min_context_size: int = 100,
        max_context_size: int = 2000,
        confidence_threshold: float = 0.8,
        probe_window_size: int = 50,
        **kwargs
    ):
        """Initialize adaptive memory."""
        super().__init__(**kwargs)
        self.min_context_size = min_context_size
        self.max_context_size = max_context_size
        self.confidence_threshold = confidence_threshold
        self.probe_window_size = probe_window_size
        
        # Component memories
        self.vector_memory = VectorStoreMemory()
        
        # Memory tracking
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.context_sizes: Dict[str, int] = {}
        self.confidence_scores: Dict[str, float] = {}
        
        # Performance tracking
        self.query_history: List[Dict[str, Any]] = []
        self.context_adjustments: List[Dict[str, Any]] = []

    async def add_memory(
        self,
        memory_id: str,
        content: str,
        initial_context_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new memory with adaptive context."""
        # Create memory entry
        memory = {
            'id': memory_id,
            'content': content,
            'created_at': datetime.now(),
            'last_accessed': datetime.now(),
            'access_count': 0,
            'metadata': metadata or {}
        }
        
        # Store memory
        self.memories[memory_id] = memory
        
        # Set initial context size
        self.context_sizes[memory_id] = (
            initial_context_size or self.min_context_size
        )
        
        # Initialize confidence score
        self.confidence_scores[memory_id] = 1.0
        
        # Add to vector memory
        await self.vector_memory.add(memory_id, content, metadata)

    async def get_memory(
        self,
        memory_id: str,
        query: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get a memory with adaptive context size."""
        if memory_id not in self.memories:
            return None
            
        memory = self.memories[memory_id]
        
        # Update access tracking
        memory['access_count'] += 1
        memory['last_accessed'] = datetime.now()
        
        # If query provided, check confidence
        if query:
            confidence = await self._probe_confidence(memory_id, query)
            self.confidence_scores[memory_id] = confidence
            
            # Adjust context size based on confidence
            await self._adjust_context_size(memory_id, confidence)
            
            # Record query
            self.query_history.append({
                'memory_id': memory_id,
                'query': query,
                'confidence': confidence,
                'context_size': self.context_sizes[memory_id],
                'timestamp': datetime.now()
            })
        
        return memory

    async def get_context_size(self, memory_id: str) -> int:
        """Get the current context size for a memory."""
        return self.context_sizes.get(memory_id, self.min_context_size)

    async def get_confidence_score(self, memory_id: str) -> float:
        """Get the current confidence score for a memory."""
        return self.confidence_scores.get(memory_id, 1.0)

    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update an existing memory."""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.update(updates)
            
            # Update vector memory
            if 'content' in updates:
                await self.vector_memory.add(memory_id, updates['content'], memory['metadata'])
            
            # Reset confidence if content changed
            if 'content' in updates:
                self.confidence_scores[memory_id] = 1.0

    async def remove_memory(self, memory_id: str) -> None:
        """Remove a memory."""
        if memory_id in self.memories:
            # Remove from vector memory
            await self.vector_memory.remove(memory_id)
            
            # Remove from tracking
            del self.memories[memory_id]
            del self.context_sizes[memory_id]
            del self.confidence_scores[memory_id]

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_memories': len(self.memories),
            'avg_context_size': np.mean(list(self.context_sizes.values())),
            'avg_confidence': np.mean(list(self.confidence_scores.values())),
            'total_queries': len(self.query_history),
            'context_adjustments': len(self.context_adjustments)
        }

    async def _probe_confidence(
        self,
        memory_id: str,
        query: str
    ) -> float:
        """Probe memory with a small context window to estimate confidence."""
        # This is a placeholder for actual confidence estimation
        # In practice, this would use the LLM to evaluate if the memory
        # is sufficient to answer the query
        return 0.8  # Placeholder

    async def _adjust_context_size(
        self,
        memory_id: str,
        confidence: float
    ) -> None:
        """Adjust context size based on confidence score."""
        current_size = self.context_sizes[memory_id]
        
        if confidence < self.confidence_threshold:
            # Increase context size
            new_size = min(
                current_size * 2,
                self.max_context_size
            )
        else:
            # Decrease context size
            new_size = max(
                current_size // 2,
                self.min_context_size
            )
        
        if new_size != current_size:
            self.context_sizes[memory_id] = new_size
            self.context_adjustments.append({
                'memory_id': memory_id,
                'old_size': current_size,
                'new_size': new_size,
                'confidence': confidence,
                'timestamp': datetime.now()
            }) 