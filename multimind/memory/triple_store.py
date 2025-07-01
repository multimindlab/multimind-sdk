import networkx as nx
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

class MemoryTripleStore:
    """
    Stores and manages (subject, predicate, object) triples using a MultiDiGraph.
    Each edge can have metadata: timestamp, score, etc.
    """
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_triple(self, subject: str, predicate: str, obj: str, score: float = 1.0, timestamp: Optional[datetime] = None, **metadata):
        """Add a triple with optional score and timestamp."""
        timestamp = timestamp or datetime.utcnow()
        self.graph.add_edge(subject, obj, key=predicate, score=score, timestamp=timestamp, **metadata)

    def remove_triple(self, subject: str, predicate: str, obj: str):
        """Remove a triple if it exists."""
        if self.graph.has_edge(subject, obj, key=predicate):
            self.graph.remove_edge(subject, obj, key=predicate)

    def update_triple(self, subject: str, predicate: str, obj: str, **updates):
        """Update metadata for a triple."""
        if self.graph.has_edge(subject, obj, key=predicate):
            edge_data = self.graph.get_edge_data(subject, obj, key=predicate)
            edge_data.update(**updates)

    def get_triples(self) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """Return all triples with metadata."""
        triples = []
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            triples.append((u, k, v, d))
        return triples

    def find_triples(self, subject: Optional[str] = None, predicate: Optional[str] = None, obj: Optional[str] = None) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """Query triples by subject, predicate, or object (wildcards allowed)."""
        results = []
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            if (subject is None or u == subject) and (predicate is None or k == predicate) and (obj is None or v == obj):
                results.append((u, k, v, d))
        return results

    def get_timeline(self, since: Optional[datetime] = None) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """Return triples added/updated since a given time."""
        timeline = []
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            ts = d.get('timestamp')
            if since is None or (ts and ts >= since):
                timeline.append((u, k, v, d))
        return sorted(timeline, key=lambda x: x[3].get('timestamp', datetime.min)) 