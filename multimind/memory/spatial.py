"""
Spatial memory implementation that manages spatial relationships and locations.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class SpatialMemory(BaseMemory):
    """Memory that manages spatial relationships and locations."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_locations: int = 1000,
        distance_threshold: float = 0.7,
        enable_analysis: bool = True,
        analysis_interval: int = 3600,  # 1 hour
        enable_relationships: bool = True,
        relationship_interval: int = 3600,  # 1 hour
        enable_clustering: bool = True,
        cluster_interval: int = 3600,  # 1 hour
        min_cluster_size: int = 3,
        enable_learning: bool = True,
        learning_rate: float = 0.1,
        enable_history: bool = True,
        history_window: int = 100,
        enable_evolution: bool = True,
        evolution_interval: int = 3600,  # 1 hour
        relationship_types: Set[str] = None,
        enable_validation: bool = True,
        validation_interval: int = 3600  # 1 hour
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_locations = max_locations
        self.distance_threshold = distance_threshold
        self.enable_analysis = enable_analysis
        self.analysis_interval = analysis_interval
        self.enable_relationships = enable_relationships
        self.relationship_interval = relationship_interval
        self.enable_clustering = enable_clustering
        self.cluster_interval = cluster_interval
        self.min_cluster_size = min_cluster_size
        self.enable_learning = enable_learning
        self.learning_rate = learning_rate
        self.enable_history = enable_history
        self.history_window = history_window
        self.enable_evolution = enable_evolution
        self.evolution_interval = evolution_interval
        self.relationship_types = relationship_types or {
            "contains",
            "contained_by",
            "adjacent_to",
            "near",
            "far_from",
            "above",
            "below",
            "inside",
            "outside"
        }
        self.enable_validation = enable_validation
        self.validation_interval = validation_interval
        
        # Initialize spatial memory storage
        self.locations: List[Dict[str, Any]] = []
        self.location_embeddings: List[List[float]] = []
        self.relationships: Dict[str, Dict[str, List[str]]] = {}  # location_id -> {relationship_type -> target_ids}
        self.clusters: Dict[str, List[str]] = {}  # cluster_id -> location_ids
        self.learning_history: Dict[str, List[Dict[str, Any]]] = {}  # location_id -> learning records
        self.location_history: List[Dict[str, Any]] = []  # Recent location updates
        self.evolution_history: Dict[str, List[Dict[str, Any]]] = {}  # location_id -> evolution records
        self.validation_history: Dict[str, List[Dict[str, Any]]] = {}  # location_id -> validation records
        self.last_analysis = datetime.now()
        self.last_relationship_update = datetime.now()
        self.last_cluster_update = datetime.now()
        self.last_evolution = datetime.now()
        self.last_validation = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message and analyze spatial information."""
        # Create new location
        location_id = f"location_{len(self.locations)}"
        new_location = {
            "id": location_id,
            "content": message["content"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "type": "message",
                "coordinates": None,
                "dimensions": None,
                "properties": {},
                "relationships": {},
                "cluster_id": None,
                "learning_progress": 0.0,
                "evolution_stage": 0,
                "validation_score": 0.0,
                "analysis_results": {}
            }
        }
        
        # Add to storage
        self.locations.append(new_location)
        
        # Get location embedding
        embedding = await self.llm.embeddings(message["content"])
        self.location_embeddings.append(embedding)
        
        # Initialize relationships
        self.relationships[location_id] = {
            rel_type: [] for rel_type in self.relationship_types
        }
        
        # Analyze spatial information
        if self.enable_analysis:
            await self._analyze_spatial_info(location_id)
        
        # Find relationships
        if self.enable_relationships:
            current_time = datetime.now()
            if (current_time - self.last_relationship_update).total_seconds() > self.relationship_interval:
                await self._find_relationships(location_id)
        
        # Update location history
        if self.enable_history:
            self.location_history.append({
                "location_id": location_id,
                "timestamp": new_location["timestamp"],
                "content": new_location["content"],
                "coordinates": new_location["metadata"]["coordinates"],
                "properties": new_location["metadata"]["properties"]
            })
            if len(self.location_history) > self.history_window:
                self.location_history.pop(0)
        
        # Update clusters
        if self.enable_clustering:
            current_time = datetime.now()
            if (current_time - self.last_cluster_update).total_seconds() > self.cluster_interval:
                await self._update_clusters()
        
        # Update learning progress
        if self.enable_learning:
            await self._update_learning_progress(location_id)
        
        # Update evolution
        if self.enable_evolution:
            current_time = datetime.now()
            if (current_time - self.last_evolution).total_seconds() > self.evolution_interval:
                await self._update_evolution(location_id)
        
        # Validate location
        if self.enable_validation:
            current_time = datetime.now()
            if (current_time - self.last_validation).total_seconds() > self.validation_interval:
                await self._validate_location(location_id)
        
        # Maintain location limit
        await self._maintain_location_limit()
        
        await self.save()

    async def _analyze_spatial_info(self, location_id: str) -> None:
        """Analyze spatial information from a message."""
        location = next(l for l in self.locations if l["id"] == location_id)
        
        try:
            # Generate analysis prompt
            prompt = f"""
            Analyze the spatial information in this message:
            
            {location['content']}
            
            Return a JSON object with:
            1. coordinates: dict with x, y, z (if available)
            2. dimensions: dict with width, height, depth (if available)
            3. properties: dict of spatial properties
            4. spatial_type: string (e.g., point, area, volume)
            """
            response = await self.llm.generate(prompt)
            analysis = json.loads(response)
            
            # Update location metadata
            location["metadata"]["coordinates"] = analysis.get("coordinates")
            location["metadata"]["dimensions"] = analysis.get("dimensions")
            location["metadata"]["properties"] = analysis.get("properties", {})
            location["metadata"]["spatial_type"] = analysis.get("spatial_type")
            location["metadata"]["analysis_results"] = analysis
            
        except Exception as e:
            print(f"Error analyzing spatial info: {e}")

    async def _find_relationships(self, location_id: str) -> None:
        """Find spatial relationships between locations."""
        location = next(l for l in self.locations if l["id"] == location_id)
        
        for other_location in self.locations:
            if other_location["id"] == location_id:
                continue
            
            # Calculate spatial similarity
            similarity = self._calculate_spatial_similarity(
                location["metadata"],
                other_location["metadata"]
            )
            
            if similarity >= self.distance_threshold:
                # Determine relationship type
                relationship_type = await self._determine_relationship_type(
                    location,
                    other_location,
                    similarity
                )
                
                if relationship_type:
                    # Add bidirectional relationship
                    self.relationships[location_id][relationship_type].append(other_location["id"])
                    self.relationships[other_location["id"]][relationship_type].append(location_id)

    def _calculate_spatial_similarity(
        self,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two spatial locations."""
        # Calculate coordinate similarity if available
        coord_similarity = 0.0
        if metadata1["coordinates"] and metadata2["coordinates"]:
            coord1 = metadata1["coordinates"]
            coord2 = metadata2["coordinates"]
            coord_similarity = 1.0 / (1.0 + sum(
                (coord1.get(k, 0) - coord2.get(k, 0)) ** 2
                for k in set(coord1.keys()) | set(coord2.keys())
            ))
        
        # Calculate dimension similarity if available
        dim_similarity = 0.0
        if metadata1["dimensions"] and metadata2["dimensions"]:
            dim1 = metadata1["dimensions"]
            dim2 = metadata2["dimensions"]
            dim_similarity = 1.0 / (1.0 + sum(
                (dim1.get(k, 0) - dim2.get(k, 0)) ** 2
                for k in set(dim1.keys()) | set(dim2.keys())
            ))
        
        # Calculate property similarity
        prop_similarity = 0.0
        if metadata1["properties"] and metadata2["properties"]:
            props1 = set(metadata1["properties"].keys())
            props2 = set(metadata2["properties"].keys())
            prop_similarity = len(props1 & props2) / len(props1 | props2) if props1 | props2 else 0
        
        return (coord_similarity + dim_similarity + prop_similarity) / 3

    async def _determine_relationship_type(
        self,
        location1: Dict[str, Any],
        location2: Dict[str, Any],
        similarity: float
    ) -> Optional[str]:
        """Determine the type of spatial relationship between two locations."""
        try:
            prompt = f"""
            Determine the spatial relationship type between these two locations:
            
            Location 1: {location1['content']}
            Coordinates: {location1['metadata']['coordinates']}
            Dimensions: {location1['metadata']['dimensions']}
            Properties: {location1['metadata']['properties']}
            
            Location 2: {location2['content']}
            Coordinates: {location2['metadata']['coordinates']}
            Dimensions: {location2['metadata']['dimensions']}
            Properties: {location2['metadata']['properties']}
            
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
        """Update clusters of related locations."""
        # Clear existing clusters
        self.clusters = {}
        
        # Group by relationship types
        for relationship_type in self.relationship_types:
            # Find connected components
            visited = set()
            
            for location_id in self.relationships:
                if location_id in visited:
                    continue
                
                # Start new cluster
                cluster_id = f"cluster_{len(self.clusters)}"
                cluster = []
                
                # DFS to find connected locations
                stack = [location_id]
                while stack:
                    current_id = stack.pop()
                    if current_id in visited:
                        continue
                    
                    visited.add(current_id)
                    cluster.append(current_id)
                    
                    # Add related locations
                    for related_id in self.relationships[current_id][relationship_type]:
                        if related_id not in visited:
                            stack.append(related_id)
                
                if len(cluster) >= self.min_cluster_size:
                    self.clusters[cluster_id] = cluster
                    
                    # Update location metadata
                    for location_id in cluster:
                        self.locations[self.locations.index(
                            next(l for l in self.locations if l["id"] == location_id)
                        )]["metadata"]["cluster_id"] = cluster_id
        
        self.last_cluster_update = datetime.now()

    async def _update_learning_progress(self, location_id: str) -> None:
        """Update learning progress for a location."""
        location = next(l for l in self.locations if l["id"] == location_id)
        
        # Calculate learning metrics
        relationship_count = sum(
            len(relationships)
            for relationships in self.relationships[location_id].values()
        )
        property_count = len(location["metadata"]["properties"])
        validation_score = location["metadata"]["validation_score"]
        
        # Update learning progress
        progress = (
            self.learning_rate * (relationship_count / len(self.relationship_types)) +
            self.learning_rate * (property_count / 10) +  # Assuming max 10 properties
            self.learning_rate * validation_score
        )
        
        location["metadata"]["learning_progress"] = min(
            1.0,
            location["metadata"]["learning_progress"] + progress
        )
        
        # Record learning update
        self.learning_history[location_id].append({
            "timestamp": datetime.now().isoformat(),
            "relationship_count": relationship_count,
            "property_count": property_count,
            "validation_score": validation_score,
            "progress": progress
        })

    async def _update_evolution(self, location_id: str) -> None:
        """Update evolution stage for a location."""
        location = next(l for l in self.locations if l["id"] == location_id)
        
        # Calculate evolution metrics
        learning_progress = location["metadata"]["learning_progress"]
        relationship_count = sum(
            len(relationships)
            for relationships in self.relationships[location_id].values()
        )
        validation_score = location["metadata"]["validation_score"]
        
        # Determine evolution stage
        if learning_progress >= 0.8 and validation_score >= 0.8:
            stage = 3  # Mature
        elif learning_progress >= 0.5 or validation_score >= 0.5:
            stage = 2  # Developing
        elif relationship_count > 0:
            stage = 1  # Emerging
        else:
            stage = 0  # New
        
        # Update evolution stage
        location["metadata"]["evolution_stage"] = stage
        
        # Record evolution
        self.evolution_history[location_id].append({
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "learning_progress": learning_progress,
            "relationship_count": relationship_count,
            "validation_score": validation_score
        })

    async def _validate_location(self, location_id: str) -> None:
        """Validate spatial information of a location."""
        location = next(l for l in self.locations if l["id"] == location_id)
        
        try:
            # Generate validation prompt
            prompt = f"""
            Validate the spatial information of this location:
            
            {location['content']}
            
            Coordinates: {location['metadata']['coordinates']}
            Dimensions: {location['metadata']['dimensions']}
            Properties: {location['metadata']['properties']}
            
            Return a JSON object with:
            1. validation_score: float (0-1)
            2. validation_reason: string
            3. inconsistencies: list of strings
            4. suggestions: list of strings
            """
            response = await self.llm.generate(prompt)
            validation = json.loads(response)
            
            # Update location metadata
            location["metadata"]["validation_score"] = validation["validation_score"]
            
            # Record validation
            self.validation_history[location_id].append({
                "timestamp": datetime.now().isoformat(),
                "score": validation["validation_score"],
                "reason": validation["validation_reason"],
                "inconsistencies": validation["inconsistencies"],
                "suggestions": validation["suggestions"]
            })
            
        except Exception as e:
            print(f"Error validating location: {e}")

    async def _maintain_location_limit(self) -> None:
        """Maintain location limit by removing least important locations."""
        if len(self.locations) > self.max_locations:
            # Sort locations by learning progress and validation score
            sorted_locations = sorted(
                self.locations,
                key=lambda x: (
                    x["metadata"]["learning_progress"] +
                    x["metadata"]["validation_score"]
                )
            )
            
            # Remove locations with lowest scores
            locations_to_remove = sorted_locations[:len(self.locations) - self.max_locations]
            for location in locations_to_remove:
                await self._remove_location(location["id"])

    async def _remove_location(self, location_id: str) -> None:
        """Remove a location and its associated data."""
        # Remove from locations
        location_idx = next(i for i, l in enumerate(self.locations) if l["id"] == location_id)
        self.locations.pop(location_idx)
        self.location_embeddings.pop(location_idx)
        
        # Remove from relationships
        if location_id in self.relationships:
            del self.relationships[location_id]
        
        # Remove from clusters
        for cluster_id, cluster in self.clusters.items():
            if location_id in cluster:
                cluster.remove(location_id)
                if len(cluster) < self.min_cluster_size:
                    del self.clusters[cluster_id]
        
        # Remove from history
        if self.enable_history:
            self.location_history = [
                l for l in self.location_history
                if l["location_id"] != location_id
            ]
        
        # Remove learning history
        if location_id in self.learning_history:
            del self.learning_history[location_id]
        
        # Remove evolution history
        if location_id in self.evolution_history:
            del self.evolution_history[location_id]
        
        # Remove validation history
        if location_id in self.validation_history:
            del self.validation_history[location_id]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from all locations."""
        messages = []
        for location in self.locations:
            messages.append({
                "role": "spatial_memory",
                "content": location["content"],
                "timestamp": location["timestamp"]
            })
        return sorted(messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear all locations."""
        self.locations = []
        self.location_embeddings = []
        self.relationships = {}
        self.clusters = {}
        self.learning_history = {}
        self.location_history = []
        self.evolution_history = {}
        self.validation_history = {}
        await self.save()

    async def save(self) -> None:
        """Save locations to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "locations": self.locations,
                    "relationships": self.relationships,
                    "clusters": self.clusters,
                    "learning_history": self.learning_history,
                    "location_history": self.location_history,
                    "evolution_history": self.evolution_history,
                    "validation_history": self.validation_history,
                    "last_analysis": self.last_analysis.isoformat(),
                    "last_relationship_update": self.last_relationship_update.isoformat(),
                    "last_cluster_update": self.last_cluster_update.isoformat(),
                    "last_evolution": self.last_evolution.isoformat(),
                    "last_validation": self.last_validation.isoformat()
                }, f)

    def load(self) -> None:
        """Load locations from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.locations = data.get("locations", [])
                self.relationships = data.get("relationships", {})
                self.clusters = data.get("clusters", {})
                self.learning_history = data.get("learning_history", {})
                self.location_history = data.get("location_history", [])
                self.evolution_history = data.get("evolution_history", {})
                self.validation_history = data.get("validation_history", {})
                self.last_analysis = datetime.fromisoformat(
                    data.get("last_analysis", datetime.now().isoformat())
                )
                self.last_relationship_update = datetime.fromisoformat(
                    data.get("last_relationship_update", datetime.now().isoformat())
                )
                self.last_cluster_update = datetime.fromisoformat(
                    data.get("last_cluster_update", datetime.now().isoformat())
                )
                self.last_evolution = datetime.fromisoformat(
                    data.get("last_evolution", datetime.now().isoformat())
                )
                self.last_validation = datetime.fromisoformat(
                    data.get("last_validation", datetime.now().isoformat())
                )
                
                # Recreate embeddings
                self.location_embeddings = []
                for location in self.locations:
                    self.location_embeddings.append(
                        self.llm.embeddings(location["content"])
                    )

    async def get_spatial_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about spatial memory."""
        stats = {
            "total_locations": len(self.locations),
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
                "average_progress": sum(
                    l["metadata"]["learning_progress"]
                    for l in self.locations
                ) / len(self.locations) if self.locations else 0,
                "locations_with_progress": sum(
                    1 for l in self.locations
                    if l["metadata"]["learning_progress"] > 0
                )
            },
            "evolution_stats": {
                "stage_distribution": {
                    stage: sum(1 for l in self.locations if l["metadata"]["evolution_stage"] == stage)
                    for stage in range(4)
                },
                "average_stage": sum(l["metadata"]["evolution_stage"] for l in self.locations) / len(self.locations) if self.locations else 0
            },
            "validation_stats": {
                "average_score": sum(
                    l["metadata"]["validation_score"]
                    for l in self.locations
                ) / len(self.locations) if self.locations else 0,
                "validated_locations": sum(
                    1 for l in self.locations
                    if l["metadata"]["validation_score"] >= 0.8
                )
            }
        }
        
        return stats

    async def get_spatial_memory_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for spatial memory optimization."""
        suggestions = []
        
        # Check location count
        if len(self.locations) > self.max_locations * 0.8:
            suggestions.append({
                "type": "location_limit",
                "suggestion": "Consider increasing max_locations or removing less important locations"
            })
        
        # Check relationship quality
        stats = await self.get_spatial_memory_stats()
        if stats["relationship_stats"]["total_relationships"] < len(self.locations) * 2:
            suggestions.append({
                "type": "relationship_development",
                "suggestion": "Consider developing more relationships between locations"
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
                "suggestion": "Consider enhancing learning mechanisms for locations"
            })
        
        # Check evolution progress
        if stats["evolution_stats"]["average_stage"] < 1.5:
            suggestions.append({
                "type": "evolution_enhancement",
                "suggestion": "Consider enhancing evolution mechanisms for locations"
            })
        
        # Check validation quality
        if stats["validation_stats"]["average_score"] < 0.8:
            suggestions.append({
                "type": "validation_improvement",
                "suggestion": "Consider improving validation mechanisms or resolving inconsistencies"
            })
        
        return suggestions 