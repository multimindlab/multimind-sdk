"""Self-evolving agent (bounded): learns few-shot exemplars from task outcomes.

Bound on "self-evolving": this does NOT rewrite its own code, tools, or
``base_prompt``. The only adaptation mechanism is which recorded
(task, response) pairs get selected as few-shot exemplars for future prompts.
Each exemplar carries a utility score that rises toward 1.0 on success and
decays toward 0.0 on failure — the same exponential-move-toward pattern used
by :class:`multimind.memory.active_learning.ActiveLearningMemory` — and only
the top ``max_examples`` by utility are ever injected into a prompt.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..models.base import BaseLLM


def _task_key(task: str) -> str:
    normalized = " ".join(task.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class Exemplar:
    """A recorded task/response pair with a running utility score."""

    task: str
    response: str
    utility: float
    success_count: int = 0
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SelfEvolvingAgent:
    """Injects the best-performing past exemplars as few-shot context.

    Args:
        model: LLM with an async ``generate(prompt) -> str`` method.
        base_prompt: fixed system/instruction prompt, never modified.
        max_examples: maximum number of exemplars injected per prompt.
        learning_rate: step size of the exponential utility update.
        initial_utility: utility assigned to a newly recorded exemplar.
        eviction_utility: exemplars whose utility decays below this are dropped.
    """

    def __init__(
        self,
        model: BaseLLM,
        base_prompt: str,
        max_examples: int = 8,
        learning_rate: float = 0.3,
        initial_utility: float = 0.5,
        eviction_utility: float = 0.05,
    ):
        self.model = model
        self.base_prompt = base_prompt
        self.max_examples = max_examples
        self.learning_rate = learning_rate
        self.initial_utility = initial_utility
        self.eviction_utility = eviction_utility
        self._exemplars: Dict[str, Exemplar] = {}

    def record_feedback(self, task: str, response: str, success: bool) -> None:
        """Record a task outcome, promoting or demoting its exemplar utility.

        Repeated feedback for the same ``task`` text updates the same
        exemplar (utility moves toward 1.0 on success, toward 0.0 on
        failure) and refreshes its stored response to the latest one.
        """
        key = _task_key(task)
        exemplar = self._exemplars.get(key)
        if exemplar is None:
            exemplar = Exemplar(task=task, response=response, utility=self.initial_utility)
            self._exemplars[key] = exemplar
        exemplar.response = response
        if success:
            exemplar.utility += self.learning_rate * (1.0 - exemplar.utility)
            exemplar.success_count += 1
        else:
            exemplar.utility -= self.learning_rate * exemplar.utility
            exemplar.failure_count += 1
        if exemplar.utility < self.eviction_utility:
            del self._exemplars[key]

    def top_exemplars(self) -> List[Exemplar]:
        """Return the ``max_examples`` exemplars with the highest utility."""
        ranked = sorted(self._exemplars.values(), key=lambda e: e.utility, reverse=True)
        return ranked[: self.max_examples]

    def build_prompt(self, task: str) -> str:
        """Assemble the base prompt, top exemplars as few-shot examples, and the task."""
        exemplars = self.top_exemplars()
        parts = [self.base_prompt]
        if exemplars:
            examples_block = "\n\n".join(
                f"Task: {ex.task}\nResponse: {ex.response}" for ex in exemplars
            )
            parts.append(f"Examples of successful past responses:\n\n{examples_block}")
        parts.append(f"Task: {task}\nResponse:")
        return "\n\n".join(parts)

    async def run(self, task: str, **kwargs) -> str:
        """Generate a response for ``task`` using the current best exemplars."""
        prompt = self.build_prompt(task)
        return await self.model.generate(prompt, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Return exemplar-pool statistics."""
        utilities = [ex.utility for ex in self._exemplars.values()]
        return {
            "total_exemplars": len(self._exemplars),
            "avg_utility": sum(utilities) / len(utilities) if utilities else 0.0,
            "max_examples": self.max_examples,
        }
