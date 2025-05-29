"""
Reinforcement-Based Memory Budgeting implementation.
"""

from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import torch
from torch import nn
from .base import BaseMemory
from .vector_store import VectorStoreMemory

class MemoryBudget:
    """Memory budget management."""
    def __init__(
        self,
        total_budget: int,
        min_budget: int,
        max_budget: int,
        decay_rate: float = 0.1
    ):
        self.total_budget = total_budget
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.decay_rate = decay_rate
        self.current_budget = total_budget
        self.last_update = datetime.now()

    def update(self, reward: float) -> None:
        """Update budget based on reward."""
        # Calculate time decay
        time_diff = (datetime.now() - self.last_update).total_seconds()
        decay = np.exp(-self.decay_rate * time_diff)
        
        # Update budget
        self.current_budget = min(
            self.max_budget,
            max(
                self.min_budget,
                self.current_budget * decay + reward
            )
        )
        self.last_update = datetime.now()

    def can_allocate(self, size: int) -> bool:
        """Check if can allocate memory."""
        return self.current_budget >= size

    def allocate(self, size: int) -> None:
        """Allocate memory."""
        if self.can_allocate(size):
            self.current_budget -= size

    def deallocate(self, size: int) -> None:
        """Deallocate memory."""
        self.current_budget = min(
            self.max_budget,
            self.current_budget + size
        )

class ReinforcementMemory(BaseMemory):
    """Memory implementation with reinforcement-based budgeting."""

    def __init__(
        self,
        total_budget: int = 1000,
        min_budget: int = 100,
        max_budget: int = 10000,
        decay_rate: float = 0.1,
        learning_rate: float = 0.01,
        discount_factor: float = 0.99,
        **kwargs
    ):
        """Initialize reinforcement memory."""
        super().__init__(**kwargs)
        
        # Budget parameters
        self.budget = MemoryBudget(
            total_budget=total_budget,
            min_budget=min_budget,
            max_budget=max_budget,
            decay_rate=decay_rate
        )
        
        # RL parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Component memories
        self.vector_memory = VectorStoreMemory()
        
        # Memory tracking
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.memory_sizes: Dict[str, int] = {}
        self.access_history: Dict[str, List[datetime]] = defaultdict(list)
        self.reward_history: List[float] = []
        
        # Q-learning components
        self.state_size = 128  # Size of state representation
        self.action_size = 3   # Keep, Remove, Compress
        self.q_network = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Statistics
        self.total_memories = 0
        self.total_rewards = 0.0
        self.optimization_rounds = 0

    async def add_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new memory with budget consideration."""
        # Calculate memory size
        memory_size = len(content.encode('utf-8'))
        
        # Check if can allocate
        if not self.budget.can_allocate(memory_size):
            # Try to free space
            await self._optimize_memory()
            
            # Check again
            if not self.budget.can_allocate(memory_size):
                raise MemoryError("Insufficient memory budget")
        
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
        self.memory_sizes[memory_id] = memory_size
        self.budget.allocate(memory_size)
        
        # Add to vector memory
        await self.vector_memory.add(memory_id, content, metadata)
        
        self.total_memories += 1

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory by ID."""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            
            # Update access tracking
            memory['access_count'] += 1
            memory['last_accessed'] = datetime.now()
            self.access_history[memory_id].append(datetime.now())
            
            # Update reward
            reward = self._calculate_reward(memory_id)
            self.budget.update(reward)
            self.reward_history.append(reward)
            self.total_rewards += reward
            
            return memory
        return None

    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update a memory with budget consideration."""
        if memory_id in self.memories:
            old_size = self.memory_sizes[memory_id]
            memory = self.memories[memory_id]
            
            # Update memory
            memory.update(updates)
            
            # Calculate new size
            new_size = len(memory['content'].encode('utf-8'))
            size_diff = new_size - old_size
            
            # Check if can allocate
            if size_diff > 0 and not self.budget.can_allocate(size_diff):
                # Try to free space
                await self._optimize_memory()
                
                # Check again
                if not self.budget.can_allocate(size_diff):
                    raise MemoryError("Insufficient memory budget")
            
            # Update budget
            if size_diff > 0:
                self.budget.allocate(size_diff)
            elif size_diff < 0:
                self.budget.deallocate(-size_diff)
            
            # Update size tracking
            self.memory_sizes[memory_id] = new_size
            
            # Update vector memory
            if 'content' in updates:
                await self.vector_memory.add(
                    memory_id,
                    updates['content'],
                    memory['metadata']
                )

    async def remove_memory(self, memory_id: str) -> None:
        """Remove a memory."""
        if memory_id in self.memories:
            # Deallocate budget
            self.budget.deallocate(self.memory_sizes[memory_id])
            
            # Remove from tracking
            del self.memories[memory_id]
            del self.memory_sizes[memory_id]
            if memory_id in self.access_history:
                del self.access_history[memory_id]
            
            # Remove from vector memory
            await self.vector_memory.remove(memory_id)

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_memories': self.total_memories,
            'current_budget': self.budget.current_budget,
            'total_rewards': self.total_rewards,
            'optimization_rounds': self.optimization_rounds,
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0.0
        }

    def _calculate_reward(self, memory_id: str) -> float:
        """Calculate reward for memory access."""
        memory = self.memories[memory_id]
        access_count = memory['access_count']
        time_since_creation = (datetime.now() - memory['created_at']).total_seconds()
        
        # Reward based on access frequency and recency
        frequency_reward = np.log1p(access_count)
        recency_reward = np.exp(-time_since_creation / 86400)  # 24-hour decay
        
        return frequency_reward * recency_reward

    async def _optimize_memory(self) -> None:
        """Optimize memory usage using reinforcement learning."""
        self.optimization_rounds += 1
        
        # Get state representation
        state = self._get_state_representation()
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(torch.FloatTensor(state))
        
        # Select action
        action = torch.argmax(q_values).item()
        
        # Apply action
        if action == 1:  # Remove
            # Remove least valuable memory
            if self.memories:
                memory_id = min(
                    self.memories.keys(),
                    key=lambda x: self._calculate_reward(x)
                )
                await self.remove_memory(memory_id)
        elif action == 2:  # Compress
            # Compress largest memory
            if self.memories:
                memory_id = max(
                    self.memories.keys(),
                    key=lambda x: self.memory_sizes[x]
                )
                await self._compress_memory(memory_id)

    def _get_state_representation(self) -> np.ndarray:
        """Get state representation for RL."""
        # Combine various metrics into state vector
        state = np.zeros(self.state_size)
        
        # Budget utilization
        state[0] = self.budget.current_budget / self.budget.max_budget
        
        # Memory count
        state[1] = len(self.memories) / self.budget.max_budget
        
        # Average access frequency
        if self.memories:
            avg_freq = np.mean([
                len(history) for history in self.access_history.values()
            ])
            state[2] = avg_freq / 100  # Normalize
        
        # Average memory size
        if self.memory_sizes:
            avg_size = np.mean(list(self.memory_sizes.values()))
            state[3] = avg_size / self.budget.max_budget
        
        return state

    async def _compress_memory(self, memory_id: str) -> None:
        """Compress a memory to save space."""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            
            # Simple compression: truncate content
            if len(memory['content']) > 100:
                memory['content'] = memory['content'][:100] + "..."
                
                # Update size tracking
                new_size = len(memory['content'].encode('utf-8'))
                size_diff = self.memory_sizes[memory_id] - new_size
                self.memory_sizes[memory_id] = new_size
                self.budget.deallocate(size_diff)
                
                # Update vector memory
                await self.vector_memory.add(
                    memory_id,
                    memory['content'],
                    memory['metadata']
                ) 