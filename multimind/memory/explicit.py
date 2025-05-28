"""
Explicit Memory implementation for storing conscious, declarative knowledge.
"""

from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
import numpy as np
from .base import BaseMemory
from .declarative import DeclarativeMemory
from .semantic import SemanticMemory

class ExplicitMemory(BaseMemory):
    """Memory implementation for conscious, declarative knowledge."""

    def __init__(
        self,
        recall_threshold: float = 0.7,
        max_facts: int = 10000,
        **kwargs
    ):
        """Initialize explicit memory."""
        super().__init__(**kwargs)
        self.recall_threshold = recall_threshold
        self.max_facts = max_facts
        
        # Component memories
        self.declarative_memory = DeclarativeMemory()
        self.semantic_memory = SemanticMemory()
        
        # Fact tracking
        self.facts: Dict[str, Dict[str, Any]] = {}
        self.fact_graph = nx.DiGraph()
        
        # Recall tracking
        self.recall_history: Dict[str, List[Dict[str, Any]]] = {}

    async def add_fact(
        self,
        fact_id: str,
        content: str,
        category: str,
        source: Optional[str] = None,
        confidence: float = 1.0,
        related_facts: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new fact with declarative knowledge."""
        # Create fact entry
        fact = {
            'id': fact_id,
            'content': content,
            'category': category,
            'source': source,
            'confidence': confidence,
            'related_facts': related_facts or [],
            'recall_count': 0,
            'last_recalled': None,
            'created_at': datetime.now(),
            'metadata': metadata or {}
        }
        
        # Store fact
        self.facts[fact_id] = fact
        
        # Add to component memories
        await self.declarative_memory.add(fact_id, content, metadata)
        await self.semantic_memory.add(fact_id, content, metadata)
        
        # Add to fact graph
        self.fact_graph.add_node(fact_id, **fact)
        
        # Add relationships
        if related_facts:
            for related_id in related_facts:
                if related_id in self.facts:
                    self.fact_graph.add_edge(fact_id, related_id)
        
        # Initialize recall history
        self.recall_history[fact_id] = []

    async def get_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """Get a fact by ID."""
        return self.facts.get(fact_id)

    async def get_facts_by_category(
        self,
        category: str,
        min_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get facts in a specific category."""
        facts = []
        for fact_id, fact in self.facts.items():
            if fact['category'] == category:
                if min_confidence is None or fact['confidence'] >= min_confidence:
                    facts.append(fact)
        return facts

    async def get_related_facts(
        self,
        fact_id: str,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Get facts related to a given fact."""
        if fact_id not in self.fact_graph:
            return []
            
        related = []
        for related_id in self.fact_graph.successors(fact_id):
            related_fact = self.facts[related_id]
            if include_metadata:
                related.append(related_fact)
            else:
                related.append({
                    'id': related_id,
                    'content': related_fact['content'],
                    'confidence': related_fact['confidence']
                })
        return related

    async def record_recall(
        self,
        fact_id: str,
        recall_score: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a recall attempt for a fact."""
        if fact_id in self.facts:
            fact = self.facts[fact_id]
            
            # Update fact
            fact['last_recalled'] = datetime.now()
            fact['recall_count'] += 1
            
            # Update confidence based on recall
            old_confidence = fact['confidence']
            recall_impact = (recall_score - old_confidence) * 0.1
            fact['confidence'] = max(0.0, min(1.0, old_confidence + recall_impact))
            
            # Record recall
            recall = {
                'timestamp': datetime.now(),
                'score': recall_score,
                'context': context or {},
                'confidence_before': old_confidence,
                'confidence_after': fact['confidence']
            }
            self.recall_history[fact_id].append(recall)

    async def get_recall_history(
        self,
        fact_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get recall history for a fact."""
        if fact_id in self.recall_history:
            history = self.recall_history[fact_id]
            if limit:
                return history[-limit:]
            return history
        return []

    async def get_fact_stats(
        self,
        fact_id: str,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get statistics for a fact."""
        if fact_id not in self.facts:
            return {}
            
        fact = self.facts[fact_id]
        history = self.recall_history[fact_id]
        
        if time_window:
            cutoff = datetime.now() - time_window
            history = [h for h in history if h['timestamp'] >= cutoff]
        
        if not history:
            return {
                'current_confidence': fact['confidence'],
                'recall_count': fact['recall_count'],
                'last_recalled': fact['last_recalled']
            }
        
        return {
            'current_confidence': fact['confidence'],
            'recall_count': fact['recall_count'],
            'last_recalled': fact['last_recalled'],
            'avg_recall_score': np.mean([h['score'] for h in history]),
            'best_recall_score': max(h['score'] for h in history),
            'confidence_change': fact['confidence'] - history[0]['confidence_before']
        }

    async def update_fact(
        self,
        fact_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update an existing fact."""
        if fact_id in self.facts:
            fact = self.facts[fact_id]
            fact.update(updates)
            
            # Update component memories
            if 'content' in updates:
                await self.declarative_memory.add(fact_id, updates['content'], fact['metadata'])
                await self.semantic_memory.add(fact_id, updates['content'], fact['metadata'])
            
            # Update graph
            self.fact_graph.nodes[fact_id].update(updates)

    async def remove_fact(self, fact_id: str) -> None:
        """Remove a fact."""
        if fact_id in self.facts:
            # Remove from component memories
            await self.declarative_memory.remove(fact_id)
            await self.semantic_memory.remove(fact_id)
            
            # Remove from graph
            self.fact_graph.remove_node(fact_id)
            
            # Remove recall history
            if fact_id in self.recall_history:
                del self.recall_history[fact_id]
            
            # Remove fact
            del self.facts[fact_id]

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_facts': len(self.facts),
            'total_categories': len(set(f['category'] for f in self.facts.values())),
            'avg_confidence': np.mean([f['confidence'] for f in self.facts.values()]),
            'total_recall_attempts': sum(len(h) for h in self.recall_history.values()),
            'fact_graph_size': self.fact_graph.number_of_nodes(),
            'fact_graph_edges': self.fact_graph.number_of_edges()
        } 