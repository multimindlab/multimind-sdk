"""
Causal Memory implementation that tracks cause-and-effect relationships between memory entries.
"""

from typing import Dict, Any, Optional, List, Set, Tuple
import networkx as nx
from datetime import datetime
from .base import BaseMemory

class CausalMemory(BaseMemory):
    """Memory implementation that tracks causal relationships between entries."""

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        max_causal_depth: int = 3,
        **kwargs
    ):
        """Initialize causal memory."""
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.max_causal_depth = max_causal_depth
        
        # Causal graph using NetworkX
        self.causal_graph = nx.DiGraph()
        
        # Storage for memory entries and their metadata
        self.storage: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        # Track causal relationships
        self.causal_links: Dict[str, List[Dict[str, Any]]] = {}
        self.confidence_scores: Dict[Tuple[str, str], float] = {}

    async def add(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        causes: Optional[List[str]] = None,
        effects: Optional[List[str]] = None
    ) -> None:
        """Add a memory entry with causal relationships."""
        self.storage[key] = value
        self.metadata[key] = metadata or {}
        
        # Initialize causal tracking
        self.causal_links[key] = []
        
        # Add to causal graph
        self.causal_graph.add_node(key)
        
        # Add causal relationships
        if causes:
            for cause in causes:
                if cause in self.storage:
                    self.causal_graph.add_edge(cause, key)
                    self.confidence_scores[(cause, key)] = 1.0
                    self.causal_links[key].append({
                        'type': 'cause',
                        'related_key': cause,
                        'confidence': 1.0,
                        'timestamp': datetime.now()
                    })
        
        if effects:
            for effect in effects:
                if effect in self.storage:
                    self.causal_graph.add_edge(key, effect)
                    self.confidence_scores[(key, effect)] = 1.0
                    self.causal_links[key].append({
                        'type': 'effect',
                        'related_key': effect,
                        'confidence': 1.0,
                        'timestamp': datetime.now()
                    })

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a memory entry."""
        return self.storage.get(key)

    async def get_causes(self, key: str, min_confidence: Optional[float] = None) -> List[str]:
        """Get all causes of a memory entry."""
        if key not in self.causal_graph:
            return []
            
        causes = list(self.causal_graph.predecessors(key))
        if min_confidence is not None:
            causes = [
                cause for cause in causes
                if self.confidence_scores.get((cause, key), 0.0) >= min_confidence
            ]
        return causes

    async def get_effects(self, key: str, min_confidence: Optional[float] = None) -> List[str]:
        """Get all effects of a memory entry."""
        if key not in self.causal_graph:
            return []
            
        effects = list(self.causal_graph.successors(key))
        if min_confidence is not None:
            effects = [
                effect for effect in effects
                if self.confidence_scores.get((key, effect), 0.0) >= min_confidence
            ]
        return effects

    async def get_causal_chain(
        self,
        start_key: str,
        end_key: str,
        max_depth: Optional[int] = None
    ) -> Optional[List[str]]:
        """Get the causal chain between two memory entries."""
        if max_depth is None:
            max_depth = self.max_causal_depth
            
        try:
            path = nx.shortest_path(
                self.causal_graph,
                source=start_key,
                target=end_key,
                cutoff=max_depth
            )
            return path
        except nx.NetworkXNoPath:
            return None

    async def update_confidence(
        self,
        cause_key: str,
        effect_key: str,
        new_confidence: float
    ) -> None:
        """Update the confidence score of a causal relationship."""
        if (cause_key, effect_key) in self.confidence_scores:
            self.confidence_scores[(cause_key, effect_key)] = new_confidence
            
            # Update causal links
            for link in self.causal_links.get(cause_key, []):
                if link['type'] == 'effect' and link['related_key'] == effect_key:
                    link['confidence'] = new_confidence
                    link['timestamp'] = datetime.now()

    async def get_causal_stats(self, key: str) -> Dict[str, Any]:
        """Get statistics about causal relationships for a memory entry."""
        if key not in self.causal_graph:
            return {}
            
        return {
            'num_causes': len(list(self.causal_graph.predecessors(key))),
            'num_effects': len(list(self.causal_graph.successors(key))),
            'avg_confidence': sum(
                self.confidence_scores.get((cause, key), 0.0)
                for cause in self.causal_graph.predecessors(key)
            ) / max(1, len(list(self.causal_graph.predecessors(key)))),
            'causal_links': self.causal_links.get(key, [])
        }

    async def remove(self, key: str) -> None:
        """Remove a memory entry and its causal relationships."""
        if key in self.storage:
            # Remove from causal graph
            self.causal_graph.remove_node(key)
            
            # Remove from storage
            del self.storage[key]
            del self.metadata[key]
            del self.causal_links[key]
            
            # Remove confidence scores
            self.confidence_scores = {
                (c, e): score
                for (c, e), score in self.confidence_scores.items()
                if c != key and e != key
            }

    async def clear(self) -> None:
        """Clear all memory entries and causal relationships."""
        self.storage.clear()
        self.metadata.clear()
        self.causal_links.clear()
        self.confidence_scores.clear()
        self.causal_graph.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """Get overall memory statistics."""
        return {
            'total_entries': len(self.storage),
            'total_causal_relationships': len(self.confidence_scores),
            'avg_confidence': sum(self.confidence_scores.values()) / max(1, len(self.confidence_scores)),
            'max_causal_depth': self.max_causal_depth,
            'confidence_threshold': self.confidence_threshold
        } 