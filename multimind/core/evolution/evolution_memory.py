from typing import Any, Dict, List, Tuple
from datetime import datetime

class EvolutionMemory:
    """
    Stores performance history of each agent-chain for learning and selection over time.
    Allows recording, retrieval, and summary of pipeline performance.
    """
    def __init__(self):
        # Key: pipeline_id (str or tuple), Value: list of (timestamp, metrics dict)
        self.history: Dict[Any, List[Tuple[datetime, Dict[str, Any]]]] = {}

    def record(self, pipeline_id: Any, metrics: Dict[str, Any]):
        """
        Record performance metrics for a given pipeline (agent-chain).
        """
        if pipeline_id not in self.history:
            self.history[pipeline_id] = []
        self.history[pipeline_id].append((datetime.utcnow(), metrics))

    def get_history(self, pipeline_id: Any) -> List[Tuple[datetime, Dict[str, Any]]]:
        """
        Retrieve the full performance history for a pipeline.
        """
        return self.history.get(pipeline_id, [])

    def summarize(self, pipeline_id: Any) -> Dict[str, Any]:
        """
        Summarize performance for a pipeline (e.g., average accuracy, cost, etc.).
        """
        records = self.get_history(pipeline_id)
        if not records:
            return {}
        summary = {}
        count = len(records)
        # Aggregate metrics
        for _, metrics in records:
            for k, v in metrics.items():
                summary[k] = summary.get(k, 0) + float(v)
        for k in summary:
            summary[k] /= count
        summary['records'] = count
        return summary 