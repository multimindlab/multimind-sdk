from typing import Any, Callable, List, Dict, Optional

class JudgeAgent:
    """
    Evaluates outputs for accuracy, quality, or fitness.
    Supports custom scoring functions, LLM-based scoring, or heuristics.
    Can be used standalone or as part of a reflexive loop.
    """
    def __init__(self, scoring_fn: Optional[Callable[[Any], float]] = None):
        """
        scoring_fn: Optional custom function to score outputs. If None, uses default heuristic.
        """
        self.scoring_fn = scoring_fn or self.default_scoring

    def default_scoring(self, output: Any) -> float:
        """
        Default heuristic: score by length (for demo). Override for real use.
        """
        return float(len(str(output)))

    def evaluate(self, outputs: List[Any]) -> List[Dict[str, Any]]:
        """
        Evaluate a list of outputs, returning a list of dicts with output and score.
        """
        results = []
        for output in outputs:
            score = self.scoring_fn(output)
            results.append({'output': output, 'score': score})
        return results

    def best(self, outputs: List[Any]) -> Any:
        """
        Return the output with the highest score.
        """
        results = self.evaluate(outputs)
        return max(results, key=lambda x: x['score'])['output'] 