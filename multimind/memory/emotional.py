"""
Emotional memory implementation that manages emotional states and responses.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class EmotionalMemory(BaseMemory):
    """Memory that manages emotional states and responses."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_states: int = 1000,
        emotion_threshold: float = 0.7,
        intensity_threshold: float = 0.5,
        enable_analysis: bool = True,
        analysis_interval: int = 3600,  # 1 hour
        enable_patterns: bool = True,
        pattern_interval: int = 3600,  # 1 hour
        enable_learning: bool = True,
        learning_rate: float = 0.1,
        emotion_weights: Dict[str, float] = None,
        enable_adaptation: bool = True,
        adaptation_rate: float = 0.05,
        enable_history: bool = True,
        history_window: int = 100,
        enable_evolution: bool = True,
        evolution_interval: int = 3600,  # 1 hour
        enable_relationships: bool = True,
        relationship_types: Set[str] = None,
        enable_clustering: bool = True,
        cluster_interval: int = 3600,  # 1 hour
        min_cluster_size: int = 3
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_states = max_states
        self.emotion_threshold = emotion_threshold
        self.intensity_threshold = intensity_threshold
        self.enable_analysis = enable_analysis
        self.analysis_interval = analysis_interval
        self.enable_patterns = enable_patterns
        self.pattern_interval = pattern_interval
        self.enable_learning = enable_learning
        self.learning_rate = learning_rate
        self.emotion_weights = emotion_weights or {
            "valence": 0.4,
            "arousal": 0.3,
            "dominance": 0.3
        }
        self.enable_adaptation = enable_adaptation
        self.adaptation_rate = adaptation_rate
        self.enable_history = enable_history
        self.history_window = history_window
        self.enable_evolution = enable_evolution
        self.evolution_interval = evolution_interval
        self.enable_relationships = enable_relationships
        self.relationship_types = relationship_types or {
            "triggers",
            "influences",
            "correlates_with",
            "opposes",
            "intensifies",
            "reduces"
        }
        self.enable_clustering = enable_clustering
        self.cluster_interval = cluster_interval
        self.min_cluster_size = min_cluster_size
        
        # Initialize emotional memory storage
        self.states: List[Dict[str, Any]] = []
        self.state_embeddings: List[List[float]] = []
        self.emotion_patterns: Dict[str, Dict[str, Any]] = {}  # pattern_id -> pattern data
        self.adaptation_history: Dict[str, List[Dict[str, Any]]] = {}  # state_id -> adaptation records
        self.learning_history: Dict[str, List[Dict[str, Any]]] = {}  # state_id -> learning records
        self.emotion_history: List[Dict[str, Any]] = []  # Recent emotion states
        self.evolution_history: Dict[str, List[Dict[str, Any]]] = {}  # state_id -> evolution records
        self.relationships: Dict[str, Dict[str, List[str]]] = {}  # state_id -> {relationship_type -> target_ids}
        self.clusters: Dict[str, List[str]] = {}  # cluster_id -> state_ids
        self.last_analysis = datetime.now()
        self.last_pattern_update = datetime.now()
        self.last_evolution = datetime.now()
        self.last_cluster_update = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message and analyze emotional state."""
        # Create new state
        state_id = f"state_{len(self.states)}"
        new_state = {
            "id": state_id,
            "content": message["content"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "type": "message",
                "emotions": {},
                "intensity": 0.0,
                "valence": 0.0,
                "arousal": 0.0,
                "dominance": 0.0,
                "pattern_matches": [],
                "learning_progress": 0.0,
                "adaptation_level": 0.0,
                "evolution_stage": 0,
                "cluster_id": None,
                "analysis_results": {}
            }
        }
        
        # Add to storage
        self.states.append(new_state)
        
        # Get state embedding
        embedding = await self.llm.embeddings(message["content"])
        self.state_embeddings.append(embedding)
        
        # Initialize relationships
        self.relationships[state_id] = {
            rel_type: [] for rel_type in self.relationship_types
        }
        
        # Analyze emotional state
        if self.enable_analysis:
            await self._analyze_emotional_state(state_id)
        
        # Find relationships
        if self.enable_relationships:
            await self._find_relationships(state_id)
        
        # Update emotion history
        if self.enable_history:
            self.emotion_history.append({
                "state_id": state_id,
                "timestamp": new_state["timestamp"],
                "emotions": new_state["metadata"]["emotions"],
                "intensity": new_state["metadata"]["intensity"]
            })
            if len(self.emotion_history) > self.history_window:
                self.emotion_history.pop(0)
        
        # Check for patterns
        if self.enable_patterns:
            current_time = datetime.now()
            if (current_time - self.last_pattern_update).total_seconds() > self.pattern_interval:
                await self._update_emotion_patterns()
        
        # Update learning progress
        if self.enable_learning:
            await self._update_learning_progress(state_id)
        
        # Update adaptation
        if self.enable_adaptation:
            await self._update_adaptation(state_id)
        
        # Update evolution
        if self.enable_evolution:
            current_time = datetime.now()
            if (current_time - self.last_evolution).total_seconds() > self.evolution_interval:
                await self._update_evolution(state_id)
        
        # Update clusters
        if self.enable_clustering:
            current_time = datetime.now()
            if (current_time - self.last_cluster_update).total_seconds() > self.cluster_interval:
                await self._update_clusters()
        
        # Maintain state limit
        await self._maintain_state_limit()
        
        await self.save()

    async def _find_relationships(self, state_id: str) -> None:
        """Find relationships between emotional states."""
        state = next(s for s in self.states if s["id"] == state_id)
        state_idx = self.states.index(state)
        
        for i, other_state in enumerate(self.states):
            if other_state["id"] == state_id:
                continue
            
            # Calculate emotional similarity
            similarity = self._calculate_emotional_similarity(
                state["metadata"],
                other_state["metadata"]
            )
            
            if similarity >= self.emotion_threshold:
                # Determine relationship type
                relationship_type = await self._determine_relationship_type(
                    state,
                    other_state,
                    similarity
                )
                
                if relationship_type:
                    # Add bidirectional relationship
                    self.relationships[state_id][relationship_type].append(other_state["id"])
                    self.relationships[other_state["id"]][relationship_type].append(state_id)

    async def _determine_relationship_type(
        self,
        state1: Dict[str, Any],
        state2: Dict[str, Any],
        similarity: float
    ) -> Optional[str]:
        """Determine the type of relationship between two emotional states."""
        try:
            prompt = f"""
            Determine the relationship type between these two emotional states:
            
            State 1: {state1['content']}
            Emotions: {state1['metadata']['emotions']}
            Intensity: {state1['metadata']['intensity']}
            
            State 2: {state2['content']}
            Emotions: {state2['metadata']['emotions']}
            Intensity: {state2['metadata']['intensity']}
            
            Similarity: {similarity}
            
            Available relationship types: {', '.join(self.relationship_types)}
            
            Return the most appropriate relationship type or 'none' if no clear relationship exists.
            """
            response = await self.llm.generate(prompt)
            
            relationship_type = response.strip().lower()
            if relationship_type in self.relationship_types:
                return relationship_type
            
            return None
            
        except Exception as e:
            print(f"Error determining relationship type: {e}")
            return None

    async def _update_clusters(self) -> None:
        """Update clusters of related emotional states."""
        # Clear existing clusters
        self.clusters = {}
        
        # Group by relationship types
        for relationship_type in self.relationship_types:
            # Find connected components
            visited = set()
            
            for state_id in self.relationships:
                if state_id in visited:
                    continue
                
                # Start new cluster
                cluster_id = f"cluster_{len(self.clusters)}"
                cluster = []
                
                # DFS to find connected states
                stack = [state_id]
                while stack:
                    current_id = stack.pop()
                    if current_id in visited:
                        continue
                    
                    visited.add(current_id)
                    cluster.append(current_id)
                    
                    # Add related states
                    for related_id in self.relationships[current_id][relationship_type]:
                        if related_id not in visited:
                            stack.append(related_id)
                
                if len(cluster) >= self.min_cluster_size:
                    self.clusters[cluster_id] = cluster
                    
                    # Update state metadata
                    for state_id in cluster:
                        self.states[self.states.index(
                            next(s for s in self.states if s["id"] == state_id)
                        )]["metadata"]["cluster_id"] = cluster_id
        
        self.last_cluster_update = datetime.now()

    async def _update_evolution(self, state_id: str) -> None:
        """Update evolution stage for an emotional state."""
        state = next(s for s in self.states if s["id"] == state_id)
        
        # Calculate evolution metrics
        adaptation_level = state["metadata"]["adaptation_level"]
        learning_progress = state["metadata"]["learning_progress"]
        relationship_count = sum(
            len(relationships)
            for relationships in self.relationships[state_id].values()
        )
        
        # Determine evolution stage
        if adaptation_level >= 0.8 and learning_progress >= 0.8:
            stage = 3  # Mature
        elif adaptation_level >= 0.5 or learning_progress >= 0.5:
            stage = 2  # Developing
        elif relationship_count > 0:
            stage = 1  # Emerging
        else:
            stage = 0  # New
        
        # Update evolution stage
        state["metadata"]["evolution_stage"] = stage
        
        # Record evolution
        self.evolution_history[state_id].append({
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "adaptation_level": adaptation_level,
            "learning_progress": learning_progress,
            "relationship_count": relationship_count
        })

    async def get_relationships(
        self,
        state_id: str,
        relationship_type: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Get relationships of an emotional state."""
        if state_id not in self.relationships:
            return {}
        
        if relationship_type:
            return {
                relationship_type: self.relationships[state_id].get(relationship_type, [])
            }
        
        return self.relationships[state_id]

    async def get_clusters(
        self,
        min_size: Optional[int] = None
    ) -> Dict[str, List[str]]:
        """Get clusters with optional size threshold."""
        if min_size is None:
            return self.clusters
        
        return {
            cluster_id: cluster
            for cluster_id, cluster in self.clusters.items()
            if len(cluster) >= min_size
        }

    async def get_evolution_history(
        self,
        state_id: str,
        min_stage: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get evolution history of an emotional state."""
        if state_id not in self.evolution_history:
            return []
        
        if min_stage is None:
            return self.evolution_history[state_id]
        
        return [
            record for record in self.evolution_history[state_id]
            if record["stage"] >= min_stage
        ]

    async def get_emotional_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about emotional memory."""
        stats = await super().get_emotional_memory_stats()
        
        # Add relationship statistics
        stats["relationship_stats"] = {
            "total_relationships": sum(
                len(relationships)
                for relationships in self.relationships.values()
            ),
            "relationship_types": {
                rel_type: sum(
                    1 for relationships in self.relationships.values()
                    if relationships[rel_type]
                )
                for rel_type in self.relationship_types
            }
        }
        
        # Add cluster statistics
        stats["cluster_stats"] = {
            "total_clusters": len(self.clusters),
            "average_cluster_size": sum(len(cluster) for cluster in self.clusters.values()) / len(self.clusters) if self.clusters else 0,
            "max_cluster_size": max(len(cluster) for cluster in self.clusters.values()) if self.clusters else 0
        }
        
        # Add evolution statistics
        stats["evolution_stats"] = {
            "stage_distribution": {
                stage: sum(1 for s in self.states if s["metadata"]["evolution_stage"] == stage)
                for stage in range(4)
            },
            "average_stage": sum(s["metadata"]["evolution_stage"] for s in self.states) / len(self.states) if self.states else 0
        }
        
        return stats

    async def get_emotional_memory_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for emotional memory optimization."""
        suggestions = await super().get_emotional_memory_suggestions()
        
        # Add relationship-related suggestions
        stats = await self.get_emotional_memory_stats()
        if stats["relationship_stats"]["total_relationships"] < len(self.states) * 2:
            suggestions.append({
                "type": "relationship_development",
                "suggestion": "Consider developing more relationships between emotional states"
            })
        
        # Add cluster-related suggestions
        if stats["cluster_stats"]["average_cluster_size"] < self.min_cluster_size:
            suggestions.append({
                "type": "cluster_development",
                "suggestion": "Consider developing more clusters or adjusting minimum cluster size"
            })
        
        # Add evolution-related suggestions
        if stats["evolution_stats"]["average_stage"] < 1.5:
            suggestions.append({
                "type": "evolution_enhancement",
                "suggestion": "Consider enhancing evolution mechanisms for emotional states"
            })
        
        return suggestions 