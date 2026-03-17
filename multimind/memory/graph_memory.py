from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from .triple_store import MemoryTripleStore
from .memory_deduplicator import MemoryDeduplicator
from .memory_scorer import MemoryScorer

class GraphMemoryAgent:
    """
    Handles (subject, predicate, object) symbolic memory graph with CRUD operations.
    Integrates deduplication and scoring.
    """
    def __init__(self):
        self.triple_store = MemoryTripleStore()
        self.deduplicator = MemoryDeduplicator()
        self.scorer = MemoryScorer()

    def add_fact(self, subject: str, predicate: str, obj: str, **metadata) -> bool:
        """Add a fact as a triple, with deduplication and scoring."""
        if self.deduplicator.is_duplicate(subject, predicate, obj, self.triple_store):
            return False
        score = self.scorer.score(subject, predicate, obj, metadata)
        self.triple_store.add_triple(subject, predicate, obj, score=score, timestamp=datetime.utcnow(), **metadata)
        return True

    def get_facts(self) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """Return all facts (triples with metadata)."""
        return self.triple_store.get_triples()

    def query(self, subject: Optional[str] = None, predicate: Optional[str] = None, obj: Optional[str] = None) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """Query facts by subject, predicate, or object (wildcards allowed)."""
        return self.triple_store.find_triples(subject, predicate, obj)

    def get_timeline(self, since: Optional[datetime] = None) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """Return facts added/updated since a given time."""
        return self.triple_store.get_timeline(since) 