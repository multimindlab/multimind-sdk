from typing import Any, Callable, Optional

class RewriterAgent:
    """
    Refines or rewrites outputs based on Judge feedback.
    Supports custom rewrite functions, LLM-based rewriting, or heuristics.
    Modular and developer-friendly for use in reflexive loops or pipelines.
    """
    def __init__(self, rewrite_fn: Optional[Callable[[Any, Optional[Any]], Any]] = None):
        """
        rewrite_fn: Optional custom function to rewrite outputs. If None, uses default heuristic.
        The function signature is (output, feedback) -> new_output.
        """
        self.rewrite_fn = rewrite_fn or self.default_rewrite

    def default_rewrite(self, output: Any, feedback: Optional[Any] = None) -> Any:
        """
        Default heuristic: append feedback to output (for demo). Override for real use.
        """
        if feedback:
            return f"{output} [Rewritten with feedback: {feedback}]"
        return f"{output} [Rewritten]"

    def rewrite(self, output: Any, feedback: Optional[Any] = None) -> Any:
        """
        Refine or rewrite the output using the rewrite function and optional feedback.
        """
        return self.rewrite_fn(output, feedback) 