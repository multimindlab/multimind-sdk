from typing import Any, Callable, List, Optional

class PlannerAgent:
    """
    Breaks high-level prompts into sub-tasks.
    Supports custom planning functions, LLM-based decomposition, or heuristics.
    Modular and developer-friendly for use in pipelines or reflexive loops.
    """
    def __init__(self, plan_fn: Optional[Callable[[str], List[str]]] = None):
        """
        plan_fn: Optional custom function to decompose prompts. If None, uses default heuristic.
        The function signature is (prompt) -> list of sub-tasks.
        """
        self.plan_fn = plan_fn or self.default_plan

    def default_plan(self, prompt: str) -> List[str]:
        """
        Default heuristic: split prompt by 'and', 'then', or commas (for demo). Override for real use.
        """
        import re
        # Split on 'and', 'then', or commas
        parts = re.split(r'\band\b|\bthen\b|,', prompt, flags=re.IGNORECASE)
        return [p.strip() for p in parts if p.strip()]

    def plan(self, prompt: str) -> List[str]:
        """
        Break the high-level prompt into sub-tasks using the planning function.
        """
        return self.plan_fn(prompt) 