"""
Fast-Weight/Hebbian Memory implementation for rapid in-context learning.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
from torch import nn
from .base import BaseMemory

class FastWeightMemory(BaseMemory):
    """Implements fast-weight memory using Hebbian learning."""
    
    def __init__(
        self,
        input_size: int = 768,
        memory_size: int = 1024,
        learning_rate: float = 0.01,
        decay_rate: float = 0.1,
        **kwargs
    ):
        """Initialize fast-weight memory."""
        super().__init__(**kwargs)
        
        # Memory parameters
        self.input_size = input_size
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        
        # Initialize weight matrix
        self.weights = torch.zeros((memory_size, input_size))
        self.usage_count = torch.zeros(memory_size)
        
        # Statistics
        self.total_updates = 0
        self.total_retrievals = 0
        self.avg_similarity = 0.0

    async def add_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add memory using Hebbian learning."""
        # Convert content to embedding
        embedding = self._get_embedding(content)
        
        # Find least used memory slot
        slot_idx = torch.argmin(self.usage_count).item()
        
        # Update weights using Hebbian learning
        self.weights[slot_idx] = (1 - self.decay_rate) * self.weights[slot_idx] + \
                                self.learning_rate * embedding
        
        # Update usage count
        self.usage_count[slot_idx] += 1
        self.total_updates += 1

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory using content-based addressing."""
        # Convert query to embedding
        query_embedding = self._get_embedding(memory_id)
        
        # Calculate similarity scores
        similarities = torch.matmul(self.weights, query_embedding)
        
        # Get most similar memory
        max_sim_idx = torch.argmax(similarities).item()
        max_similarity = similarities[max_sim_idx].item()
        
        # Update statistics
        self.total_retrievals += 1
        self.avg_similarity = (self.avg_similarity * (self.total_retrievals - 1) + 
                             max_similarity) / self.total_retrievals
        
        if max_similarity > 0.5:  # Similarity threshold
            return {
                'id': memory_id,
                'content': self._decode_embedding(self.weights[max_sim_idx]),
                'similarity': max_similarity,
                'usage_count': self.usage_count[max_sim_idx].item()
            }
        return None

    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update memory using Hebbian learning."""
        if 'content' in updates:
            # Convert new content to embedding
            new_embedding = self._get_embedding(updates['content'])
            
            # Find existing memory
            query_embedding = self._get_embedding(memory_id)
            similarities = torch.matmul(self.weights, query_embedding)
            max_sim_idx = torch.argmax(similarities).item()
            
            # Update weights
            self.weights[max_sim_idx] = (1 - self.decay_rate) * self.weights[max_sim_idx] + \
                                       self.learning_rate * new_embedding

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_updates': self.total_updates,
            'total_retrievals': self.total_retrievals,
            'avg_similarity': self.avg_similarity,
            'memory_utilization': (self.usage_count > 0).float().mean().item(),
            'avg_usage_count': self.usage_count.mean().item()
        }

    def _get_embedding(self, text: str) -> torch.Tensor:
        """Convert text to embedding."""
        # This would typically use a pre-trained model
        # For now, we'll use a simple random projection
        return torch.randn(self.input_size)

    def _decode_embedding(self, embedding: torch.Tensor) -> str:
        """Convert embedding back to text."""
        # This would typically use a decoder model
        # For now, we'll return a placeholder
        return f"Memory content with similarity {embedding.norm().item():.2f}" 