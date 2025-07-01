from typing import List, Dict, Any

class MultiObjectiveJudgeAgent:
    """
    Scores outputs by multiple dimensions (accuracy, cost, speed, creativity).
    For demo, expects each output to be a dict with these keys and returns the one with the highest total score.
    """
    def __init__(self, weights: Dict[str, float] = None):
        # Weights for each objective (default: equal)
        self.weights = weights or {'accuracy': 1.0, 'cost': 1.0, 'speed': 1.0, 'creativity': 1.0}

    def score(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Scores each output and returns the one with the highest weighted sum.
        """
        def weighted_sum(output):
            return sum(self.weights.get(k, 0) * float(output.get(k, 0)) for k in self.weights)
        best = max(outputs, key=weighted_sum)
        return best 