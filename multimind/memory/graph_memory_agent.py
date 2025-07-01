from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from .triple_store import MemoryTripleStore
from .memory_deduplicator import MemoryDeduplicator
from .memory_scorer import MemoryScorer
from .memory_merge_engine import MemoryMergeEngine

class GraphMemoryAgent:
    """
    Advanced symbolic memory agent for (subject, predicate, object) triples.
    Integrates deduplication, scoring, merging, and timeline. Supports backend switching.
    Modular and developer-friendly for use in agentic workflows.
    """
    def __init__(self, backend: Optional[str] = 'networkx',
                 deduplicator: Optional[MemoryDeduplicator] = None,
                 scorer: Optional[MemoryScorer] = None,
                 merger: Optional[MemoryMergeEngine] = None):
        # For now, only networkx backend is implemented; extend for Neo4j, etc.
        self.backend = backend
        self.triple_store = MemoryTripleStore()  # Could swap for Neo4jTripleStore, etc.
        self.deduplicator = deduplicator or MemoryDeduplicator()
        self.scorer = scorer or MemoryScorer()
        self.merger = merger or MemoryMergeEngine()

    def add_fact(self, subject: str, predicate: str, obj: str, **metadata) -> bool:
        """
        Add a fact as a triple, with deduplication, scoring, and timeline.
        If duplicate or contradictory, can merge or reject.
        """
        if self.deduplicator.is_duplicate(subject, predicate, obj, self.triple_store):
            # Optionally merge with existing
            for u, p, o, meta in self.triple_store.find_triples(subject, predicate, obj):
                merged = self.merger.merge(meta, metadata)
                self.triple_store.update_triple(u, p, o, **merged)
            return False
        if self.deduplicator.is_contradictory(subject, predicate, obj, self.triple_store):
            # Optionally reject or handle contradiction
            return False
        score = self.scorer.score(subject, predicate, obj, metadata)
        self.triple_store.add_triple(subject, predicate, obj, score=score, timestamp=datetime.utcnow(), **metadata)
        self.scorer.update_frequency(subject, predicate, obj)
        return True

    def get_facts(self) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """
        Return all facts (triples with metadata).
        """
        return self.triple_store.get_triples()

    def query(self, subject: Optional[str] = None, predicate: Optional[str] = None, obj: Optional[str] = None) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """
        Query facts by subject, predicate, or object (wildcards allowed).
        """
        return self.triple_store.find_triples(subject, predicate, obj)

    def get_timeline(self, since: Optional[datetime] = None) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """
        Return facts added/updated since a given time.
        """
        return self.triple_store.get_timeline(since)

    def switch_backend(self, backend: str):
        """
        Switch the underlying triple store backend (e.g., networkx, Neo4j).
        For now, only networkx is implemented.
        """
        # Placeholder for backend switching logic
        if backend == 'networkx':
            self.triple_store = MemoryTripleStore()
        # elif backend == 'neo4j':
        #     self.triple_store = Neo4jTripleStore(...)
        else:
            raise NotImplementedError(f"Backend '{backend}' not supported yet.")
        self.backend = backend 