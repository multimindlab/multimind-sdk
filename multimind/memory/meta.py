"""
Meta-Memory implementation that tracks memory usage statistics and importance scores.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
from .base import BaseMemory

class MetaMemory(BaseMemory):
    """Memory that tracks usage statistics and importance scores for each memory entry."""

    def __init__(
        self,
        base_memory: BaseMemory,
        importance_decay: float = 0.95,
        success_threshold: float = 0.7,
        max_history: int = 1000,
        **kwargs
    ):
        """Initialize MetaMemory with a base memory implementation."""
        super().__init__(**kwargs)
        self.base_memory = base_memory
        self.importance_decay = importance_decay
        self.success_threshold = success_threshold
        self.max_history = max_history
        
        # Track usage statistics
        self.usage_history: Dict[str, List[Dict[str, Any]]] = {}
        self.importance_scores: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        self.success_rates: Dict[str, float] = {}

    async def add(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a memory entry with initial usage statistics."""
        await self.base_memory.add(key, value, metadata)
        
        # Initialize statistics
        self.usage_history[key] = []
        self.importance_scores[key] = 1.0
        self.access_counts[key] = 0
        self.success_rates[key] = 0.0

    async def get(self, key: str) -> Any:
        """Retrieve a memory entry and update its usage statistics."""
        value = await self.base_memory.get(key)
        
        # Update access count
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        
        # Record access
        self.usage_history[key].append({
            'timestamp': datetime.now(),
            'type': 'read',
            'success': value is not None
        })
        
        # Trim history if needed
        if len(self.usage_history[key]) > self.max_history:
            self.usage_history[key] = self.usage_history[key][-self.max_history:]
        
        return value

    async def update_importance(self, key: str, success: bool) -> None:
        """Update the importance score based on usage success."""
        if key not in self.importance_scores:
            return
            
        # Update success rate
        history = self.usage_history.get(key, [])
        if history:
            success_count = sum(1 for entry in history if entry.get('success', False))
            self.success_rates[key] = success_count / len(history)
        
        # Update importance score
        current_score = self.importance_scores[key]
        if success:
            # Increase importance for successful uses
            new_score = current_score * (1 + (1 - self.importance_decay))
        else:
            # Decrease importance for unsuccessful uses
            new_score = current_score * self.importance_decay
        
        self.importance_scores[key] = min(1.0, max(0.0, new_score))

    async def get_important_memories(self, threshold: float = 0.5) -> List[str]:
        """Get keys of memories with importance scores above the threshold."""
        return [
            key for key, score in self.importance_scores.items()
            if score >= threshold
        ]

    async def get_frequently_accessed(self, min_accesses: int = 5) -> List[str]:
        """Get keys of frequently accessed memories."""
        return [
            key for key, count in self.access_counts.items()
            if count >= min_accesses
        ]

    async def get_successful_memories(self, threshold: float = 0.7) -> List[str]:
        """Get keys of memories with high success rates."""
        return [
            key for key, rate in self.success_rates.items()
            if rate >= threshold
        ]

    async def get_memory_stats(self, key: str) -> Dict[str, Any]:
        """Get detailed statistics for a memory entry."""
        return {
            'importance_score': self.importance_scores.get(key, 0.0),
            'access_count': self.access_counts.get(key, 0),
            'success_rate': self.success_rates.get(key, 0.0),
            'usage_history': self.usage_history.get(key, []),
            'last_accessed': self.usage_history[key][-1]['timestamp'] if key in self.usage_history and self.usage_history[key] else None
        }

    async def cleanup(self, min_importance: float = 0.1) -> None:
        """Remove memories with low importance scores."""
        keys_to_remove = [
            key for key, score in self.importance_scores.items()
            if score < min_importance
        ]
        
        for key in keys_to_remove:
            await self.base_memory.remove(key)
            del self.usage_history[key]
            del self.importance_scores[key]
            del self.access_counts[key]
            del self.success_rates[key] 