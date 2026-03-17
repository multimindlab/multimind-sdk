from typing import Any, Callable, List, Optional

class ThinkerAgent:
    """
    Performs strategic planning and reflection.
    Integrates with memory (e.g., GraphMemoryAgent).
    Supports custom reasoning functions, chain-of-thought, or LLM-based logic.
    Modular and developer-friendly for use in pipelines or reflexive loops.
    """
    def __init__(self, memory=None, reason_fn: Optional[Callable[[str, Any], List[str]]] = None):
        """
        memory: Optional memory module (e.g., GraphMemoryAgent) for context.
        reason_fn: Optional custom function for reasoning. If None, uses default heuristic.
        The function signature is (goal, memory) -> list of reasoning steps.
        """
        self.memory = memory
        self.reason_fn = reason_fn or self.default_reason

    def default_reason(self, goal: str, memory: Any = None) -> List[str]:
        """
        Default heuristic: if memory is present, query for facts related to the goal and chain as steps.
        Otherwise, return a generic plan.
        """
        steps = []
        if self.memory and hasattr(self.memory, 'query'):
            facts = self.memory.query(predicate='related_to', obj=goal)
            for subj, pred, obj, meta in facts:
                steps.append(f"Step: Use {subj} because it is {pred} {obj}")
        if not steps:
            steps.append(f"No direct memory found for goal '{goal}'. Try researching or asking for more info.")
        return steps

    def think(self, goal: str) -> List[str]:
        """
        Perform strategic planning and reflection for the given goal.
        Returns a list of reasoning steps.
        """
        return self.reason_fn(goal, self.memory) 