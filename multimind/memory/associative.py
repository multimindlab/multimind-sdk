"""
Associative memory implementation that stores and retrieves information based on associations and patterns.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class AssociativeMemory(BaseMemory):
    """Memory that stores and retrieves information based on associations and patterns."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_associations: int = 1000,
        similarity_threshold: float = 0.7,
        pattern_threshold: float = 0.8,
        enable_patterns: bool = True,
        pattern_interval: int = 3600,  # 1 hour
        enable_relationships: bool = True,
        relationship_types: Set[str] = None,
        enable_clustering: bool = True,
        cluster_interval: int = 3600,  # 1 hour
        min_cluster_size: int = 3,
        enable_learning: bool = True,
        learning_rate: float = 0.1,
        enable_temporal: bool = True,
        temporal_window: int = 3600,  # 1 hour
        enable_confidence: bool = True,
        confidence_threshold: float = 0.7,
        enable_analysis: bool = True,
        analysis_interval: int = 3600,  # 1 hour
        enable_evolution: bool = True,
        evolution_interval: int = 3600  # 1 hour
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_associations = max_associations
        self.similarity_threshold = similarity_threshold
        self.pattern_threshold = pattern_threshold
        self.enable_patterns = enable_patterns
        self.pattern_interval = pattern_interval
        self.enable_relationships = enable_relationships
        self.relationship_types = relationship_types or {
            "similar_to",
            "related_to",
            "part_of",
            "causes",
            "influences",
            "depends_on",
            "precedes",
            "follows",
            "contradicts",
            "supports"
        }
        self.enable_clustering = enable_clustering
        self.cluster_interval = cluster_interval
        self.min_cluster_size = min_cluster_size
        self.enable_learning = enable_learning
        self.learning_rate = learning_rate
        self.enable_temporal = enable_temporal
        self.temporal_window = temporal_window
        self.enable_confidence = enable_confidence
        self.confidence_threshold = confidence_threshold
        self.enable_analysis = enable_analysis
        self.analysis_interval = analysis_interval
        self.enable_evolution = enable_evolution
        self.evolution_interval = evolution_interval
        
        # Initialize associative memory storage
        self.associations: List[Dict[str, Any]] = []
        self.association_embeddings: List[List[float]] = []
        self.patterns: Dict[str, Dict[str, Any]] = {}  # pattern_id -> pattern data
        self.relationships: Dict[str, Dict[str, List[str]]] = {}  # association_id -> {relationship_type -> target_ids}
        self.clusters: Dict[str, List[str]] = {}  # cluster_id -> association_ids
        self.learning_history: Dict[str, List[Dict[str, Any]]] = {}  # association_id -> learning records
        self.temporal_relationships: Dict[str, List[Dict[str, Any]]] = {}  # association_id -> temporal records
        self.confidence_scores: Dict[str, float] = {}  # association_id -> confidence score
        self.evolution_history: Dict[str, List[Dict[str, Any]]] = {}  # association_id -> evolution records
        self.last_pattern_update = datetime.now()
        self.last_cluster_update = datetime.now()
        self.last_analysis = datetime.now()
        self.last_evolution = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message as new association."""
        # Create new association
        association_id = f"assoc_{len(self.associations)}"
        new_association = {
            "id": association_id,
            "content": message["content"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "type": "message",
                "confidence": 1.0,
                "pattern_matches": [],
                "cluster_id": None,
                "learning_progress": 0.0,
                "temporal_links": [],
                "evolution_stage": 0,
                "analysis_results": {}
            }
        }
        
        # Add to storage
        self.associations.append(new_association)
        
        # Get association embedding
        embedding = await self.llm.embeddings(message["content"])
        self.association_embeddings.append(embedding)
        
        # Initialize relationships
        self.relationships[association_id] = {
            rel_type: [] for rel_type in self.relationship_types
        }
        
        # Find relationships
        if self.enable_relationships:
            await self._find_relationships(association_id)
        
        # Update temporal relationships
        if self.enable_temporal:
            await self._update_temporal_relationships(association_id)
        
        # Update confidence scores
        if self.enable_confidence:
            await self._update_confidence_scores(association_id)
        
        # Check for patterns
        if self.enable_patterns:
            current_time = datetime.now()
            if (current_time - self.last_pattern_update).total_seconds() > self.pattern_interval:
                await self._update_patterns()
        
        # Check for clustering
        if self.enable_clustering:
            current_time = datetime.now()
            if (current_time - self.last_cluster_update).total_seconds() > self.cluster_interval:
                await self._update_clusters()
        
        # Update learning progress
        if self.enable_learning:
            await self._update_learning_progress(association_id)
        
        # Update evolution
        if self.enable_evolution:
            current_time = datetime.now()
            if (current_time - self.last_evolution).total_seconds() > self.evolution_interval:
                await self._update_evolution(association_id)
        
        # Maintain association limit
        await self._maintain_association_limit()
        
        await self.save()

    async def _find_relationships(self, association_id: str) -> None:
        """Find relationships between associations."""
        association = next(a for a in self.associations if a["id"] == association_id)
        association_idx = self.associations.index(association)
        
        for i, other_association in enumerate(self.associations):
            if other_association["id"] == association_id:
                continue
            
            # Calculate similarity
            similarity = self._cosine_similarity(
                self.association_embeddings[association_idx],
                self.association_embeddings[i]
            )
            
            if similarity >= self.similarity_threshold:
                # Determine relationship type
                relationship_type = await self._determine_relationship_type(
                    association,
                    other_association,
                    similarity
                )
                
                if relationship_type:
                    # Add bidirectional relationship
                    self.relationships[association_id][relationship_type].append(other_association["id"])
                    self.relationships[other_association["id"]][relationship_type].append(association_id)

    async def _determine_relationship_type(
        self,
        assoc1: Dict[str, Any],
        assoc2: Dict[str, Any],
        similarity: float
    ) -> Optional[str]:
        """Determine the type of relationship between two associations."""
        try:
            prompt = f"""
            Determine the relationship type between these two pieces of information:
            
            Information 1: {assoc1['content']}
            Information 2: {assoc2['content']}
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

    async def _update_patterns(self) -> None:
        """Update patterns in associations."""
        # Group similar associations
        groups = []
        used_indices = set()
        
        for i, assoc1 in enumerate(self.associations):
            if i in used_indices:
                continue
            
            group = [i]
            used_indices.add(i)
            
            for j, assoc2 in enumerate(self.associations[i+1:], i+1):
                if j in used_indices:
                    continue
                
                similarity = self._cosine_similarity(
                    self.association_embeddings[i],
                    self.association_embeddings[j]
                )
                
                if similarity >= self.pattern_threshold:
                    group.append(j)
                    used_indices.add(j)
            
            if len(group) >= self.min_cluster_size:
                groups.append(group)
        
        # Create patterns from groups
        for group in groups:
            pattern_id = f"pattern_{len(self.patterns)}"
            
            # Extract common elements
            common_elements = await self._extract_common_elements([
                self.associations[i] for i in group
            ])
            
            # Create pattern
            self.patterns[pattern_id] = {
                "id": pattern_id,
                "associations": [self.associations[i]["id"] for i in group],
                "common_elements": common_elements,
                "confidence": len(group) / len(self.associations),
                "timestamp": datetime.now().isoformat()
            }
            
            # Update association metadata
            for i in group:
                self.associations[i]["metadata"]["pattern_matches"].append(pattern_id)
        
        self.last_pattern_update = datetime.now()

    async def _extract_common_elements(self, associations: List[Dict[str, Any]]) -> List[str]:
        """Extract common elements from a group of associations."""
        try:
            prompt = f"""
            Extract common elements or patterns from these pieces of information:
            
            {chr(10).join(f'Information {i+1}: {assoc["content"]}' for i, assoc in enumerate(associations))}
            
            Return a list of common elements, one per line.
            """
            response = await self.llm.generate(prompt)
            
            return [line.strip() for line in response.split('\n') if line.strip()]
            
        except Exception as e:
            print(f"Error extracting common elements: {e}")
            return []

    async def _update_clusters(self) -> None:
        """Update clusters of related associations."""
        # Clear existing clusters
        self.clusters = {}
        
        # Group by relationship types
        for relationship_type in self.relationship_types:
            # Find connected components
            visited = set()
            
            for assoc_id in self.relationships:
                if assoc_id in visited:
                    continue
                
                # Start new cluster
                cluster_id = f"cluster_{len(self.clusters)}"
                cluster = []
                
                # DFS to find connected associations
                stack = [assoc_id]
                while stack:
                    current_id = stack.pop()
                    if current_id in visited:
                        continue
                    
                    visited.add(current_id)
                    cluster.append(current_id)
                    
                    # Add related associations
                    for related_id in self.relationships[current_id][relationship_type]:
                        if related_id not in visited:
                            stack.append(related_id)
                
                if len(cluster) >= self.min_cluster_size:
                    self.clusters[cluster_id] = cluster
                    
                    # Update association metadata
                    for assoc_id in cluster:
                        self.associations[self.associations.index(
                            next(a for a in self.associations if a["id"] == assoc_id)
                        )]["metadata"]["cluster_id"] = cluster_id
        
        self.last_cluster_update = datetime.now()

    async def _update_learning_progress(self, association_id: str) -> None:
        """Update learning progress for an association."""
        association = next(a for a in self.associations if a["id"] == association_id)
        
        # Calculate learning metrics
        relationship_count = sum(
            len(relationships)
            for relationships in self.relationships[association_id].values()
        )
        pattern_matches = len(association["metadata"]["pattern_matches"])
        cluster_membership = 1 if association["metadata"]["cluster_id"] else 0
        
        # Update learning progress
        progress = (
            self.learning_rate * (relationship_count / len(self.relationship_types)) +
            self.learning_rate * (pattern_matches / len(self.patterns)) +
            self.learning_rate * cluster_membership
        )
        
        association["metadata"]["learning_progress"] = min(
            1.0,
            association["metadata"]["learning_progress"] + progress
        )
        
        # Record learning update
        self.learning_history[association_id].append({
            "timestamp": datetime.now().isoformat(),
            "relationship_count": relationship_count,
            "pattern_matches": pattern_matches,
            "cluster_membership": cluster_membership,
            "progress": progress
        })

    async def _maintain_association_limit(self) -> None:
        """Maintain association limit by removing least important associations."""
        if len(self.associations) > self.max_associations:
            # Sort associations by learning progress
            sorted_associations = sorted(
                self.associations,
                key=lambda x: x["metadata"]["learning_progress"]
            )
            
            # Remove associations with lowest progress
            associations_to_remove = sorted_associations[:len(self.associations) - self.max_associations]
            for association in associations_to_remove:
                await self._remove_association(association["id"])

    async def _remove_association(self, association_id: str) -> None:
        """Remove an association and its relationships."""
        # Remove from associations
        association_idx = next(i for i, a in enumerate(self.associations) if a["id"] == association_id)
        self.associations.pop(association_idx)
        self.association_embeddings.pop(association_idx)
        
        # Remove relationships
        if association_id in self.relationships:
            del self.relationships[association_id]
        
        # Remove from patterns
        for pattern in self.patterns.values():
            if association_id in pattern["associations"]:
                pattern["associations"].remove(association_id)
        
        # Remove from clusters
        for cluster in self.clusters.values():
            if association_id in cluster:
                cluster.remove(association_id)
        
        # Remove learning history
        if association_id in self.learning_history:
            del self.learning_history[association_id]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from all associations."""
        messages = []
        for association in self.associations:
            messages.append({
                "role": "associative_memory",
                "content": association["content"],
                "timestamp": association["timestamp"]
            })
        return sorted(messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear all associations."""
        self.associations = []
        self.association_embeddings = []
        self.patterns = {}
        self.relationships = {}
        self.clusters = {}
        self.learning_history = {}
        await self.save()

    async def save(self) -> None:
        """Save associations to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "associations": self.associations,
                    "patterns": self.patterns,
                    "relationships": self.relationships,
                    "clusters": self.clusters,
                    "learning_history": self.learning_history,
                    "temporal_relationships": self.temporal_relationships,
                    "confidence_scores": self.confidence_scores,
                    "evolution_history": self.evolution_history,
                    "last_pattern_update": self.last_pattern_update.isoformat(),
                    "last_cluster_update": self.last_cluster_update.isoformat(),
                    "last_analysis": self.last_analysis.isoformat(),
                    "last_evolution": self.last_evolution.isoformat()
                }, f)

    def load(self) -> None:
        """Load associations from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.associations = data.get("associations", [])
                self.patterns = data.get("patterns", {})
                self.relationships = data.get("relationships", {})
                self.clusters = data.get("clusters", {})
                self.learning_history = data.get("learning_history", {})
                self.temporal_relationships = data.get("temporal_relationships", {})
                self.confidence_scores = data.get("confidence_scores", {})
                self.evolution_history = data.get("evolution_history", {})
                self.last_pattern_update = datetime.fromisoformat(
                    data.get("last_pattern_update", datetime.now().isoformat())
                )
                self.last_cluster_update = datetime.fromisoformat(
                    data.get("last_cluster_update", datetime.now().isoformat())
                )
                self.last_analysis = datetime.fromisoformat(
                    data.get("last_analysis", datetime.now().isoformat())
                )
                self.last_evolution = datetime.fromisoformat(
                    data.get("last_evolution", datetime.now().isoformat())
                )
                
                # Recreate embeddings
                self.association_embeddings = []
                for association in self.associations:
                    self.association_embeddings.append(
                        self.llm.embeddings(association["content"])
                    )

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2)

    async def get_association_by_id(self, association_id: str) -> Optional[Dict[str, Any]]:
        """Get an association by its ID."""
        try:
            return next(a for a in self.associations if a["id"] == association_id)
        except StopIteration:
            return None

    async def get_relationships(
        self,
        association_id: str,
        relationship_type: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Get relationships of an association."""
        if association_id not in self.relationships:
            return {}
        
        if relationship_type:
            return {
                relationship_type: self.relationships[association_id].get(relationship_type, [])
            }
        
        return self.relationships[association_id]

    async def get_patterns(
        self,
        min_confidence: Optional[float] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get patterns with optional confidence threshold."""
        if min_confidence is None:
            return self.patterns
        
        return {
            pattern_id: pattern
            for pattern_id, pattern in self.patterns.items()
            if pattern["confidence"] >= min_confidence
        }

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

    async def get_learning_history(
        self,
        association_id: str,
        min_progress: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get learning history of an association."""
        if association_id not in self.learning_history:
            return []
        
        if min_progress is None:
            return self.learning_history[association_id]
        
        return [
            record for record in self.learning_history[association_id]
            if record["progress"] >= min_progress
        ]

    async def get_temporal_relationships(
        self,
        association_id: str,
        relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get temporal relationships of an association."""
        if association_id not in self.temporal_relationships:
            return []
        
        if relationship_type:
            return [
                link for link in self.temporal_relationships[association_id]
                if link["relationship"] == relationship_type
            ]
        
        return self.temporal_relationships[association_id]

    async def get_confidence_score(self, association_id: str) -> float:
        """Get confidence score for an association."""
        return self.confidence_scores.get(association_id, 0.0)

    async def get_evolution_history(
        self,
        association_id: str,
        min_stage: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get evolution history of an association."""
        if association_id not in self.evolution_history:
            return []
        
        if min_stage is None:
            return self.evolution_history[association_id]
        
        return [
            record for record in self.evolution_history[association_id]
            if record["stage"] >= min_stage
        ]

    async def get_associative_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about associative memory."""
        stats = {
            "total_associations": len(self.associations),
            "pattern_stats": {
                "total_patterns": len(self.patterns),
                "average_confidence": sum(p["confidence"] for p in self.patterns.values()) / len(self.patterns) if self.patterns else 0,
                "average_pattern_size": sum(len(p["associations"]) for p in self.patterns.values()) / len(self.patterns) if self.patterns else 0
            },
            "relationship_stats": {
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
            },
            "cluster_stats": {
                "total_clusters": len(self.clusters),
                "average_cluster_size": sum(len(cluster) for cluster in self.clusters.values()) / len(self.clusters) if self.clusters else 0,
                "max_cluster_size": max(len(cluster) for cluster in self.clusters.values()) if self.clusters else 0
            },
            "learning_stats": {
                "average_progress": sum(a["metadata"]["learning_progress"] for a in self.associations) / len(self.associations) if self.associations else 0,
                "associations_with_progress": sum(1 for a in self.associations if a["metadata"]["learning_progress"] > 0)
            },
            "temporal_stats": {
                "total_temporal_links": sum(
                    len(links)
                    for links in self.temporal_relationships.values()
                ),
                "average_links_per_association": sum(
                    len(links)
                    for links in self.temporal_relationships.values()
                ) / len(self.temporal_relationships) if self.temporal_relationships else 0
            },
            "confidence_stats": {
                "average_confidence": sum(self.confidence_scores.values()) / len(self.confidence_scores) if self.confidence_scores else 0,
                "high_confidence_associations": sum(1 for score in self.confidence_scores.values() if score >= self.confidence_threshold)
            },
            "evolution_stats": {
                "stage_distribution": {
                    stage: sum(1 for a in self.associations if a["metadata"]["evolution_stage"] == stage)
                    for stage in range(4)
                },
                "average_stage": sum(a["metadata"]["evolution_stage"] for a in self.associations) / len(self.associations) if self.associations else 0
            }
        }
        
        return stats

    async def get_associative_memory_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for associative memory optimization."""
        suggestions = []
        
        # Check association count
        if len(self.associations) > self.max_associations * 0.8:
            suggestions.append({
                "type": "association_limit",
                "suggestion": "Consider increasing max_associations or removing less important associations"
            })
        
        # Check pattern quality
        stats = await self.get_associative_memory_stats()
        if stats["pattern_stats"]["average_confidence"] < 0.7:
            suggestions.append({
                "type": "pattern_quality",
                "suggestion": "Consider adjusting pattern threshold or improving pattern extraction"
            })
        
        # Check relationship distribution
        if stats["relationship_stats"]["total_relationships"] < len(self.associations) * 2:
            suggestions.append({
                "type": "relationship_development",
                "suggestion": "Consider developing more relationships between associations"
            })
        
        # Check cluster quality
        if stats["cluster_stats"]["average_cluster_size"] < self.min_cluster_size:
            suggestions.append({
                "type": "cluster_development",
                "suggestion": "Consider developing more clusters or adjusting minimum cluster size"
            })
        
        # Check learning progress
        if stats["learning_stats"]["average_progress"] < 0.5:
            suggestions.append({
                "type": "learning_enhancement",
                "suggestion": "Consider enhancing learning mechanisms for associations"
            })
        
        # Add temporal-related suggestions
        if stats["temporal_stats"]["average_links_per_association"] < 2:
            suggestions.append({
                "type": "temporal_development",
                "suggestion": "Consider developing more temporal relationships between associations"
            })
        
        # Add confidence-related suggestions
        if stats["confidence_stats"]["high_confidence_associations"] < len(self.associations) * 0.3:
            suggestions.append({
                "type": "confidence_improvement",
                "suggestion": "Consider improving confidence scoring or relationship development"
            })
        
        # Add evolution-related suggestions
        if stats["evolution_stats"]["average_stage"] < 1.5:
            suggestions.append({
                "type": "evolution_enhancement",
                "suggestion": "Consider enhancing evolution mechanisms for associations"
            })
        
        return suggestions 