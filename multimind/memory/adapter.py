"""
Adapter-Based Session Memory implementation.
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
from torch import nn
from .base import BaseMemory

class AdapterLayer(nn.Module):
    """Adapter layer for fine-tuning."""
    def __init__(self, input_size: int, adapter_size: int):
        super().__init__()
        self.down = nn.Linear(input_size, adapter_size)
        self.up = nn.Linear(adapter_size, input_size)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(x)))

class AdapterMemory(BaseMemory):
    """Implements adapter-based session memory."""
    
    def __init__(
        self,
        input_size: int = 768,
        adapter_size: int = 64,
        learning_rate: float = 0.001,
        **kwargs
    ):
        """Initialize adapter memory."""
        super().__init__(**kwargs)
        
        # Memory parameters
        self.input_size = input_size
        self.adapter_size = adapter_size
        self.learning_rate = learning_rate
        
        # Initialize adapter
        self.adapter = AdapterLayer(input_size, adapter_size)
        self.optimizer = torch.optim.Adam(self.adapter.parameters(), lr=learning_rate)
        
        # Session tracking
        self.session_memories: Dict[str, List[torch.Tensor]] = {}
        self.session_adapters: Dict[str, AdapterLayer] = {}
        
        # Statistics
        self.total_sessions = 0
        self.total_updates = 0
        self.avg_adaptation_loss = 0.0

    async def add_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add memory and adapt session."""
        # Get session ID from metadata or generate new
        session_id = metadata.get('session_id', f'session_{self.total_sessions}')
        
        # Convert content to embedding
        embedding = self._get_embedding(content)
        
        # Initialize session if new
        if session_id not in self.session_memories:
            self.session_memories[session_id] = []
            self.session_adapters[session_id] = AdapterLayer(
                self.input_size,
                self.adapter_size
            )
            self.total_sessions += 1
        
        # Store memory
        self.session_memories[session_id].append(embedding)
        
        # Adapt session
        await self._adapt_session(session_id)

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory using session-adapted embeddings."""
        # Convert query to embedding
        query_embedding = self._get_embedding(memory_id)
        
        best_similarity = 0.0
        best_memory = None
        best_session = None
        
        # Search across all sessions
        for session_id, memories in self.session_memories.items():
            adapter = self.session_adapters[session_id]
            
            # Adapt query to session
            adapted_query = adapter(query_embedding)
            
            # Find most similar memory
            for memory in memories:
                similarity = torch.cosine_similarity(
                    adapted_query.unsqueeze(0),
                    memory.unsqueeze(0)
                ).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_memory = memory
                    best_session = session_id
        
        if best_similarity > 0.5:  # Similarity threshold
            return {
                'id': memory_id,
                'content': self._decode_embedding(best_memory),
                'session_id': best_session,
                'similarity': best_similarity
            }
        return None

    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update memory and adapt session."""
        if 'content' in updates and 'session_id' in updates:
            session_id = updates['session_id']
            
            if session_id in self.session_memories:
                # Convert new content to embedding
                new_embedding = self._get_embedding(updates['content'])
                
                # Find most similar memory in session
                query_embedding = self._get_embedding(memory_id)
                adapter = self.session_adapters[session_id]
                adapted_query = adapter(query_embedding)
                
                similarities = [
                    torch.cosine_similarity(
                        adapted_query.unsqueeze(0),
                        memory.unsqueeze(0)
                    ).item()
                    for memory in self.session_memories[session_id]
                ]
                
                if similarities:
                    max_idx = max(range(len(similarities)), key=lambda i: similarities[i])
                    self.session_memories[session_id][max_idx] = new_embedding
                    
                    # Re-adapt session
                    await self._adapt_session(session_id)

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_sessions': self.total_sessions,
            'total_updates': self.total_updates,
            'avg_adaptation_loss': self.avg_adaptation_loss,
            'sessions': {
                session_id: len(memories)
                for session_id, memories in self.session_memories.items()
            }
        }

    async def _adapt_session(self, session_id: str) -> None:
        """Adapt session using stored memories."""
        memories = self.session_memories[session_id]
        adapter = self.session_adapters[session_id]
        
        if len(memories) > 1:
            # Create pairs for adaptation
            for i in range(len(memories) - 1):
                source = memories[i]
                target = memories[i + 1]
                
                # Forward pass
                adapted = adapter(source)
                loss = torch.nn.functional.mse_loss(adapted, target)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                self.total_updates += 1
                self.avg_adaptation_loss = (
                    self.avg_adaptation_loss * (self.total_updates - 1) +
                    loss.item()
                ) / self.total_updates

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