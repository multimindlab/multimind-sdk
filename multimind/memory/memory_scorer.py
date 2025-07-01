from datetime import datetime
from typing import Any, Callable, Dict, Optional

class MemoryScorer:
    """
    Assigns relevance scores to memory entries for injection or decay.
    Supports pluggable scoring strategies: recency, frequency, importance, LLM-based, etc.
    """
    def __init__(self, strategy: Optional[str] = 'recency', custom_scorer: Optional[Callable] = None):
        self.strategy = strategy
        self.custom_scorer = custom_scorer
        self.frequency_map: Dict[Any, int] = {}  # For frequency-based scoring

    def score(self, subject, predicate, obj, metadata):
        """
        Assign a score to the triple using the selected strategy.
        """
        if self.custom_scorer:
            return self.custom_scorer(subject, predicate, obj, metadata)
        if self.strategy == 'recency':
            now = datetime.utcnow().timestamp()
            ts = metadata.get('timestamp')
            if ts is None:
                return 1.0
            age = max(1, now - ts.timestamp())
            return 1.0 / age
        elif self.strategy == 'frequency':
            key = (subject, predicate, obj)
            freq = self.frequency_map.get(key, 1)
            return float(freq)
        elif self.strategy == 'importance':
            return float(metadata.get('importance', 1.0))
        elif self.strategy == 'llm':
            # Placeholder: integrate with LLM for semantic scoring
            return float(metadata.get('llm_score', 1.0))
        else:
            return 1.0

    def update_frequency(self, subject, predicate, obj):
        """
        Update frequency count for a triple (call this when accessed/added).
        """
        key = (subject, predicate, obj)
        self.frequency_map[key] = self.frequency_map.get(key, 0) + 1 