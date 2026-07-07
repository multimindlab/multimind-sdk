"""
Planning memory: plan-step recording and success-weighted next-step suggestion.

Honest core: given a current state, the next action is suggested purely from
the observed success/failure frequency of actions previously taken in
similar historical states (same style as
:class:`multimind.memory.reinforcement.ReinforcementMemory` — real
statistics over recorded history, no fabricated model of the environment).

The previous version of this module simulated multi-step "rollouts" via a
dummy ``_simulate_action`` that just echoed the action back as a fake
"outcome", and a dummy ``_is_goal_reached`` that string-matched a "status"
field. Without a real environment/simulator, predicting the outcome of a
hypothetical future action can't be done honestly, so that code (and the two
component-memory fields it wired in — one of which, ``VectorStoreMemory()``,
would not even instantiate without a required ``llm`` argument) has been
removed rather than kept as fabricated output. Multi-step planning reduces
to repeated single-step suggestion: callers execute the suggested action,
observe the real outcome, record it, and ask again.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseMemory


class PlanningMemory(BaseMemory):
    """Memory implementation with plan-step recording and next-step suggestion."""

    def __init__(
        self,
        memory_key: str = "chat_history",
        similarity_threshold: float = 0.8,
        storage_path: Optional[str] = None,
        **kwargs,
    ):
        """Initialize planning memory."""
        super().__init__(memory_key)
        self.similarity_threshold = similarity_threshold
        self.storage_path = Path(storage_path) if storage_path else None
        self.kwargs = kwargs

        # Memory tracking
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.plans: Dict[str, Dict[str, Any]] = {}

        # Performance tracking
        self.plan_success: Dict[str, List[bool]] = defaultdict(list)

    async def add_memory(
        self,
        memory_id: str,
        content: str,
        state: Optional[Dict[str, Any]] = None,
        action: Optional[str] = None,
        outcome: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a new memory with planning context."""
        memory = {
            "id": memory_id,
            "content": content,
            "state": state or {},
            "action": action,
            "outcome": outcome or {},
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "access_count": 0,
            "metadata": metadata or {},
        }

        self.memories[memory_id] = memory

        # If this is a state-action-outcome memory, add to plans
        if state and action and outcome:
            self.plans[memory_id] = {
                "state": state,
                "action": action,
                "outcome": outcome,
                "success": outcome.get("success", True),
            }

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory by ID."""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory["access_count"] += 1
            memory["last_accessed"] = datetime.now().isoformat()
            return memory
        return None

    async def suggest_next_action(
        self, current_state: Dict[str, Any], min_similarity: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Suggest the next action from success-weighted history of similar states.

        Ranks actions previously taken in states similar to ``current_state``
        by their observed success rate (ties broken by higher support).
        Returns ``None`` when no similar historical state/action data exists.
        """
        similar_memories = await self._find_similar_states(current_state, min_similarity)

        action_outcomes: Dict[str, List[bool]] = defaultdict(list)
        for memory in similar_memories:
            action = memory.get("action")
            if action:
                action_outcomes[action].append(
                    bool(memory.get("outcome", {}).get("success", False))
                )

        if not action_outcomes:
            return None

        def score(action: str) -> Tuple[float, int]:
            outcomes = action_outcomes[action]
            return (fmean(outcomes), len(outcomes))

        best_action = max(action_outcomes, key=score)
        outcomes = action_outcomes[best_action]
        return {
            "action": best_action,
            "success_rate": fmean(outcomes),
            "support": len(outcomes),
        }

    async def record_plan_outcome(
        self, plan_id: str, success: bool, actual_outcome: Dict[str, Any]
    ) -> None:
        """Record the outcome of a plan execution."""
        self.plan_success[plan_id].append(success)

        if plan_id in self.plans:
            self.plans[plan_id]["outcome"] = actual_outcome
            self.plans[plan_id]["success"] = success

    async def get_similar_plans(
        self, state: Dict[str, Any], min_similarity: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get plans similar to the given state."""
        similar_plans = []
        for plan in self.plans.values():
            similarity = self._calculate_state_similarity(state, plan["state"])
            if min_similarity is None or similarity >= min_similarity:
                plan_copy = plan.copy()
                plan_copy["similarity"] = similarity
                similar_plans.append(plan_copy)
        return similar_plans

    async def get_plan_stats(self, plan_id: str) -> Dict[str, Any]:
        """Get statistics for a plan."""
        if plan_id not in self.plans:
            return {}

        successes = self.plan_success[plan_id]
        return {
            "success_rate": fmean(successes) if successes else 0.0,
            "total_executions": len(successes),
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        per_plan_rates = [fmean(s) for s in self.plan_success.values() if s]
        return {
            "total_memories": len(self.memories),
            "total_plans": len(self.plans),
            "avg_success_rate": fmean(per_plan_rates) if per_plan_rates else 0.0,
        }

    async def _find_similar_states(
        self, state: Dict[str, Any], min_similarity: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Find memories with similar states."""
        threshold = min_similarity if min_similarity is not None else self.similarity_threshold
        similar_memories = []
        for memory in self.memories.values():
            if memory["state"]:
                similarity = self._calculate_state_similarity(state, memory["state"])
                if similarity >= threshold:
                    similar_memories.append(memory)
        return similar_memories

    @staticmethod
    def _calculate_state_similarity(state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
        """Jaccard-style overlap: matching key/value pairs over the union of keys."""
        keys = set(state1) | set(state2)
        if not keys:
            return 1.0
        matches = sum(1 for key in keys if state1.get(key) == state2.get(key))
        return matches / len(keys)

    # --- BaseMemory interface ---

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add a message as a plain memory (no planning context)."""
        memory_id = f"message_{len(self.memories)}"
        await self.add_memory(
            memory_id, message["content"], metadata={"role": message.get("role", "user")}
        )

    async def get_messages(self) -> List[Dict[str, str]]:
        """Get all stored memories as messages, oldest first."""
        ordered = sorted(self.memories.values(), key=lambda m: m["created_at"])
        return [
            {
                "role": memory["metadata"].get("role", "planning_memory"),
                "content": memory["content"],
            }
            for memory in ordered
        ]

    async def clear(self) -> None:
        """Clear all memories and plans."""
        self.memories.clear()
        self.plans.clear()
        self.plan_success.clear()
        await self.save()

    async def save(self) -> None:
        """Save memories and plans to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump(
                    {
                        "memories": self.memories,
                        "plans": self.plans,
                        "plan_success": dict(self.plan_success),
                    },
                    f,
                )

    async def load(self) -> None:
        """Load memories and plans from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path) as f:
                data = json.load(f)
            self.memories = data.get("memories", {})
            self.plans = data.get("plans", {})
            self.plan_success = defaultdict(list, data.get("plan_success", {}))
