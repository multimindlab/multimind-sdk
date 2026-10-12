"""
Reinforcement-Based Memory Budgeting implementation.

Uses pure-python tabular Q-learning (no torch) over a discretized budget state
to decide between keeping, evicting, or compressing memories when the budget
runs low.
"""

import json
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseMemory


class MemoryBudget:
    """Memory budget management."""

    def __init__(
        self, total_budget: int, min_budget: int, max_budget: int, decay_rate: float = 0.1
    ):
        self.total_budget = total_budget
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.decay_rate = decay_rate
        self.current_budget = total_budget
        self.last_update = datetime.now()

    def update(self, reward: float) -> None:
        """Update budget based on reward."""
        time_diff = (datetime.now() - self.last_update).total_seconds()
        decay = math.exp(-self.decay_rate * time_diff)
        self.current_budget = min(
            self.max_budget, max(self.min_budget, self.current_budget * decay + reward)
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
        self.current_budget = min(self.max_budget, self.current_budget + size)


# Actions for the Q-learning policy
ACTION_KEEP = 0
ACTION_REMOVE = 1
ACTION_COMPRESS = 2
_ACTIONS = (ACTION_KEEP, ACTION_REMOVE, ACTION_COMPRESS)


class ReinforcementMemory(BaseMemory):
    """Memory implementation with reinforcement-based budgeting (tabular Q-learning)."""

    def __init__(
        self,
        total_budget: int = 1000,
        min_budget: int = 100,
        max_budget: int = 10000,
        decay_rate: float = 0.1,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        storage_path: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Initialize reinforcement memory."""
        super().__init__(**kwargs)

        # Budget parameters
        self.budget = MemoryBudget(
            total_budget=total_budget,
            min_budget=min_budget,
            max_budget=max_budget,
            decay_rate=decay_rate,
        )

        # Q-learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        self.q_table: Dict[Tuple[int, int, int], List[float]] = defaultdict(
            lambda: [0.0] * len(_ACTIONS)
        )

        self.storage_path = Path(storage_path) if storage_path else None

        # Memory tracking
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.memory_sizes: Dict[str, int] = {}
        self.access_history: Dict[str, List[datetime]] = defaultdict(list)
        self.reward_history: List[float] = []

        # Statistics
        self.total_memories = 0
        self.total_rewards = 0.0
        self.optimization_rounds = 0

    async def add_memory(
        self, memory_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new memory with budget consideration."""
        memory_size = len(content.encode("utf-8"))

        if not self.budget.can_allocate(memory_size):
            await self._optimize_memory()
            if not self.budget.can_allocate(memory_size):
                raise MemoryError("Insufficient memory budget")

        memory = {
            "id": memory_id,
            "content": content,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "access_count": 0,
            "metadata": metadata or {},
        }

        self.memories[memory_id] = memory
        self.memory_sizes[memory_id] = memory_size
        self.budget.allocate(memory_size)
        self.total_memories += 1

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory by ID."""
        if memory_id in self.memories:
            memory = self.memories[memory_id]

            memory["access_count"] += 1
            memory["last_accessed"] = datetime.now()
            self.access_history[memory_id].append(datetime.now())

            reward = self._calculate_reward(memory_id)
            self.budget.update(reward)
            self.reward_history.append(reward)
            self.total_rewards += reward

            return memory
        return None

    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> None:
        """Update a memory with budget consideration."""
        if memory_id in self.memories:
            old_size = self.memory_sizes[memory_id]
            memory = self.memories[memory_id]

            memory.update(updates)

            new_size = len(memory["content"].encode("utf-8"))
            size_diff = new_size - old_size

            if size_diff > 0 and not self.budget.can_allocate(size_diff):
                await self._optimize_memory()
                if not self.budget.can_allocate(size_diff):
                    raise MemoryError("Insufficient memory budget")

            if size_diff > 0:
                self.budget.allocate(size_diff)
            elif size_diff < 0:
                self.budget.deallocate(-size_diff)

            self.memory_sizes[memory_id] = new_size

    async def remove_memory(self, memory_id: str) -> None:
        """Remove a memory."""
        if memory_id in self.memories:
            self.budget.deallocate(self.memory_sizes[memory_id])
            del self.memories[memory_id]
            del self.memory_sizes[memory_id]
            if memory_id in self.access_history:
                del self.access_history[memory_id]

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_memories": self.total_memories,
            "current_budget": self.budget.current_budget,
            "total_rewards": self.total_rewards,
            "optimization_rounds": self.optimization_rounds,
            "avg_reward": fmean(self.reward_history) if self.reward_history else 0.0,
            "q_table_size": len(self.q_table),
        }

    def _calculate_reward(self, memory_id: str) -> float:
        """Calculate reward for memory access."""
        memory = self.memories[memory_id]
        access_count = memory["access_count"]
        time_since_creation = (datetime.now() - memory["created_at"]).total_seconds()

        frequency_reward = math.log1p(access_count)
        recency_reward = math.exp(-time_since_creation / 86400)  # 24-hour decay

        return frequency_reward * recency_reward

    async def _optimize_memory(self) -> None:
        """Optimize memory usage with one tabular Q-learning step."""
        self.optimization_rounds += 1

        state = self._get_state()
        action = self._select_action(state)

        freed = 0
        if action == ACTION_REMOVE and self.memories:
            memory_id = min(self.memories.keys(), key=lambda x: self._calculate_reward(x))
            freed = self.memory_sizes[memory_id]
            await self.remove_memory(memory_id)
        elif action == ACTION_COMPRESS and self.memories:
            memory_id = max(self.memories.keys(), key=lambda x: self.memory_sizes[x])
            before = self.memory_sizes[memory_id]
            await self._compress_memory(memory_id)
            freed = before - self.memory_sizes[memory_id]

        # Reward: fraction of max budget freed; keeping frees nothing and is
        # penalized when the budget is under pressure.
        reward = freed / self.budget.max_budget
        if action == ACTION_KEEP and self.budget.current_budget < self.budget.min_budget * 2:
            reward = -0.1

        next_state = self._get_state()
        self._update_q(state, action, reward, next_state)

    def _get_state(self) -> Tuple[int, int, int]:
        """Discretize budget/memory metrics into a Q-table state."""
        utilization_bucket = min(
            9, int(10 * (1 - self.budget.current_budget / self.budget.max_budget))
        )
        count_bucket = min(9, len(self.memories) // 10)
        if self.memory_sizes:
            avg_size = fmean(self.memory_sizes.values())
            size_bucket = min(9, int(10 * avg_size / self.budget.max_budget))
        else:
            size_bucket = 0
        return (utilization_bucket, count_bucket, size_bucket)

    def _select_action(self, state: Tuple[int, int, int]) -> int:
        """Epsilon-greedy action selection."""
        if self._rng.random() < self.epsilon:
            return self._rng.choice(_ACTIONS)
        q_values = self.q_table[state]
        return max(_ACTIONS, key=lambda a: q_values[a])

    def _update_q(
        self,
        state: Tuple[int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int],
    ) -> None:
        """Standard Q-learning update."""
        q_values = self.q_table[state]
        best_next = max(self.q_table[next_state])
        q_values[action] += self.learning_rate * (
            reward + self.discount_factor * best_next - q_values[action]
        )

    async def _compress_memory(self, memory_id: str) -> None:
        """Compress a memory to save space."""
        if memory_id in self.memories:
            memory = self.memories[memory_id]

            if len(memory["content"]) > 100:
                memory["content"] = memory["content"][:100] + "..."

                new_size = len(memory["content"].encode("utf-8"))
                size_diff = self.memory_sizes[memory_id] - new_size
                self.memory_sizes[memory_id] = new_size
                self.budget.deallocate(size_diff)

    # --- BaseMemory interface ---

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add a message as a budgeted memory."""
        memory_id = f"message_{self.total_memories}"
        await self.add_memory(memory_id, message["content"], {"role": message.get("role", "user")})

    async def get_messages(self) -> List[Dict[str, str]]:
        """Get all stored memories as messages."""
        return [
            {
                "role": memory["metadata"].get("role", "reinforcement_memory"),
                "content": memory["content"],
            }
            for memory in self.memories.values()
        ]

    async def clear(self) -> None:
        """Clear all memories and learning state."""
        for memory_id in list(self.memories):
            await self.remove_memory(memory_id)
        self.q_table.clear()
        self.reward_history = []
        await self.save()

    async def save(self) -> None:
        """Save memories and Q-table to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump(
                    {
                        "memories": {
                            memory_id: {
                                "id": memory["id"],
                                "content": memory["content"],
                                "metadata": memory["metadata"],
                                "access_count": memory["access_count"],
                            }
                            for memory_id, memory in self.memories.items()
                        },
                        "q_table": {
                            ",".join(map(str, state)): q_values
                            for state, q_values in self.q_table.items()
                        },
                    },
                    f,
                )

    async def load(self) -> None:
        """Load memories and Q-table from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path) as f:
                data = json.load(f)
            for memory_id, memory in data.get("memories", {}).items():
                content = memory["content"]
                self.memories[memory_id] = {
                    "id": memory["id"],
                    "content": content,
                    "created_at": datetime.now(),
                    "last_accessed": datetime.now(),
                    "access_count": memory.get("access_count", 0),
                    "metadata": memory.get("metadata", {}),
                }
                self.memory_sizes[memory_id] = len(content.encode("utf-8"))
            for state_key, q_values in data.get("q_table", {}).items():
                state = tuple(int(part) for part in state_key.split(","))
                self.q_table[state] = q_values
