"""
Generative Memory implementation for periodic memory regeneration and reconstruction.
"""

import json
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from statistics import fmean
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .base import BaseMemory


class GenerativeMemory(BaseMemory):
    """Memory implementation for generative replay and reconstruction."""

    def __init__(
        self,
        regeneration_interval: timedelta = timedelta(days=7),
        reconstruction_threshold: float = 0.8,
        max_memories: int = 10000,
        storage_path: Optional[str] = None,
        vector_memory: Optional[BaseMemory] = None,
        semantic_memory: Optional[BaseMemory] = None,
        **kwargs,
    ):
        """Initialize generative memory."""
        super().__init__(**kwargs)
        self.regeneration_interval = regeneration_interval
        self.reconstruction_threshold = reconstruction_threshold
        self.max_memories = max_memories
        self.storage_path = Path(storage_path) if storage_path else None

        # Optional component memories (must expose async add/remove)
        self.vector_memory = vector_memory
        self.semantic_memory = semantic_memory

        # Memory tracking
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.regeneration_history: Dict[str, List[Dict[str, Any]]] = {}

        # Reconstruction tracking
        self.reconstruction_scores: Dict[str, float] = {}

    async def _propagate_add(
        self, memory_id: str, content: str, metadata: Optional[Dict[str, Any]]
    ) -> None:
        if self.vector_memory is not None:
            await self.vector_memory.add(memory_id, content, metadata)
        if self.semantic_memory is not None:
            await self.semantic_memory.add(memory_id, content, metadata)

    async def add_memory(
        self,
        memory_id: str,
        content: str,
        category: str,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a new memory with generative capabilities."""
        # Create memory entry
        memory = {
            "id": memory_id,
            "content": content,
            "category": category,
            "source": source,
            "original_content": content,
            "last_regenerated": datetime.now(),
            "regeneration_count": 0,
            "created_at": datetime.now(),
            "metadata": metadata or {},
        }

        # Store memory
        self.memories[memory_id] = memory
        await self._propagate_add(memory_id, content, metadata)

        # Initialize regeneration history
        self.regeneration_history[memory_id] = []

        # Initialize reconstruction score
        self.reconstruction_scores[memory_id] = 1.0

    async def regenerate_memory(
        self, memory_id: str, new_content: str, confidence: float = 1.0
    ) -> None:
        """Regenerate a memory with new content."""
        if memory_id in self.memories:
            memory = self.memories[memory_id]

            # Record regeneration
            regeneration = {
                "timestamp": datetime.now(),
                "old_content": memory["content"],
                "new_content": new_content,
                "confidence": confidence,
            }
            self.regeneration_history[memory_id].append(regeneration)

            # Update memory
            memory["content"] = new_content
            memory["last_regenerated"] = datetime.now()
            memory["regeneration_count"] += 1
            await self._propagate_add(memory_id, new_content, memory["metadata"])

            # Update reconstruction score
            self.reconstruction_scores[memory_id] = confidence

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory by ID."""
        return self.memories.get(memory_id)

    async def get_memories_by_category(
        self, category: str, min_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get memories in a specific category."""
        memories = []
        for memory_id, memory in self.memories.items():
            if memory["category"] == category:
                if (
                    min_confidence is None
                    or self.reconstruction_scores[memory_id] >= min_confidence
                ):
                    memories.append(memory)
        return memories

    async def get_regeneration_history(
        self, memory_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get regeneration history for a memory."""
        if memory_id in self.regeneration_history:
            history = self.regeneration_history[memory_id]
            if limit:
                return history[-limit:]
            return history
        return []

    async def check_regeneration_needed(self, memory_id: str) -> bool:
        """Check if a memory needs regeneration."""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            time_since_regeneration = datetime.now() - memory["last_regenerated"]
            return time_since_regeneration >= self.regeneration_interval
        return False

    async def get_memories_needing_regeneration(self) -> List[str]:
        """Get IDs of all memories due for regeneration."""
        due = []
        for memory_id in self.memories:
            if await self.check_regeneration_needed(memory_id):
                due.append(memory_id)
        return due

    async def regenerate_due_memories(
        self,
        regenerate_fn: Callable[[Dict[str, Any]], Awaitable[Tuple[str, float]]],
    ) -> List[str]:
        """
        Regenerate all due memories using a caller-supplied async generator.

        regenerate_fn receives the memory dict and must return
        (new_content, confidence). Returns the list of regenerated memory IDs.
        """
        regenerated = []
        for memory_id in await self.get_memories_needing_regeneration():
            new_content, confidence = await regenerate_fn(self.memories[memory_id])
            await self.regenerate_memory(memory_id, new_content, confidence)
            regenerated.append(memory_id)
        return regenerated

    async def get_memory_stats(
        self, memory_id: str, time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get statistics for a memory."""
        if memory_id not in self.memories:
            return {}

        memory = self.memories[memory_id]
        history = self.regeneration_history[memory_id]

        if time_window:
            cutoff = datetime.now() - time_window
            history = [h for h in history if h["timestamp"] >= cutoff]

        if not history:
            return {
                "regeneration_count": memory["regeneration_count"],
                "last_regenerated": memory["last_regenerated"],
                "reconstruction_score": self.reconstruction_scores[memory_id],
            }

        return {
            "regeneration_count": memory["regeneration_count"],
            "last_regenerated": memory["last_regenerated"],
            "reconstruction_score": self.reconstruction_scores[memory_id],
            "avg_confidence": fmean([h["confidence"] for h in history]),
            "content_drift": self._calculate_content_drift(
                memory["original_content"], memory["content"]
            ),
        }

    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> None:
        """Update an existing memory."""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.update(updates)

            # Update component memories
            if "content" in updates:
                await self._propagate_add(memory_id, updates["content"], memory["metadata"])

    async def remove_memory(self, memory_id: str) -> None:
        """Remove a memory."""
        if memory_id in self.memories:
            # Remove from component memories
            if self.vector_memory is not None:
                await self.vector_memory.remove(memory_id)
            if self.semantic_memory is not None:
                await self.semantic_memory.remove(memory_id)

            # Remove regeneration history
            if memory_id in self.regeneration_history:
                del self.regeneration_history[memory_id]

            # Remove reconstruction score
            if memory_id in self.reconstruction_scores:
                del self.reconstruction_scores[memory_id]

            # Remove memory
            del self.memories[memory_id]

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_memories": len(self.memories),
            "total_categories": len(set(m["category"] for m in self.memories.values())),
            "avg_regeneration_count": (
                fmean([m["regeneration_count"] for m in self.memories.values()])
                if self.memories
                else 0.0
            ),
            "total_regenerations": sum(len(h) for h in self.regeneration_history.values()),
            "avg_reconstruction_score": (
                fmean(self.reconstruction_scores.values()) if self.reconstruction_scores else 0.0
            ),
        }

    def _calculate_content_drift(self, original: str, current: str) -> float:
        """Calculate the lexical drift between original and current content."""
        if original == current:
            return 0.0
        return 1.0 - SequenceMatcher(None, original, current).ratio()

    # --- BaseMemory interface ---

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add a message as a generative memory."""
        memory_id = f"message_{len(self.memories)}"
        await self.add_memory(
            memory_id,
            message["content"],
            category="message",
            metadata={"role": message.get("role", "user")},
        )

    async def get_messages(self) -> List[Dict[str, str]]:
        """Get all stored memories as messages."""
        return [
            {
                "role": memory["metadata"].get("role", "generative_memory"),
                "content": memory["content"],
            }
            for memory in self.memories.values()
        ]

    async def clear(self) -> None:
        """Clear all memories and regeneration state."""
        self.memories = {}
        self.regeneration_history = {}
        self.reconstruction_scores = {}
        await self.save()

    async def save(self) -> None:
        """Save memories to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = {
                memory_id: {
                    "id": memory["id"],
                    "content": memory["content"],
                    "category": memory["category"],
                    "source": memory["source"],
                    "original_content": memory["original_content"],
                    "last_regenerated": memory["last_regenerated"].isoformat(),
                    "regeneration_count": memory["regeneration_count"],
                    "created_at": memory["created_at"].isoformat(),
                    "metadata": memory["metadata"],
                }
                for memory_id, memory in self.memories.items()
            }
            with open(self.storage_path, "w") as f:
                json.dump(
                    {
                        "memories": serializable,
                        "reconstruction_scores": self.reconstruction_scores,
                    },
                    f,
                )

    async def load(self) -> None:
        """Load memories from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path) as f:
                data = json.load(f)
            for memory_id, memory in data.get("memories", {}).items():
                self.memories[memory_id] = {
                    **memory,
                    "last_regenerated": datetime.fromisoformat(memory["last_regenerated"]),
                    "created_at": datetime.fromisoformat(memory["created_at"]),
                }
                self.regeneration_history.setdefault(memory_id, [])
            self.reconstruction_scores.update(data.get("reconstruction_scores", {}))
