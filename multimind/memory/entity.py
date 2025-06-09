"""
Entity-based memory implementation for tracking entities and their relationships.
"""

from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from .base import BaseMemory

class EntityMemory(BaseMemory):
    """Memory that tracks entities and their relationships."""

    def __init__(
        self,
        entity_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        max_entities: Optional[int] = None,
        max_relationships: Optional[int] = None,
        **kwargs
    ):
        """Initialize entity memory."""
        super().__init__(**kwargs)
        
        # Configuration
        self.entity_types = set(entity_types or ["person", "organization", "location", "concept"])
        self.relationship_types = set(relationship_types or ["related_to", "part_of", "located_in", "works_for"])
        self.max_entities = max_entities
        self.max_relationships = max_relationships
        
        # Storage
        self.entities: Dict[str, Dict[str, Any]] = {}  # entity_id -> entity_data
        self.relationships: Dict[str, Set[str]] = {}  # entity_id -> set of related entity_ids
        self.entity_metadata: Dict[str, Dict[str, Any]] = {}  # entity_id -> metadata
        self.relationship_metadata: Dict[tuple, Dict[str, Any]] = {}  # (entity1_id, entity2_id) -> metadata

    async def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an entity to memory."""
        if entity_type not in self.entity_types:
            raise ValueError(f"Invalid entity type: {entity_type}")
            
        # Create entity
        self.entities[entity_id] = {
            "type": entity_type,
            "properties": properties,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        # Add metadata
        self.entity_metadata[entity_id] = metadata or {}
        
        # Initialize relationships
        if entity_id not in self.relationships:
            self.relationships[entity_id] = set()
            
        # Check limits
        if self.max_entities and len(self.entities) > self.max_entities:
            await self._prune_entities()

    async def add_relationship(
        self,
        entity1_id: str,
        entity2_id: str,
        relationship_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a relationship between entities."""
        if relationship_type not in self.relationship_types:
            raise ValueError(f"Invalid relationship type: {relationship_type}")
            
        if entity1_id not in self.entities or entity2_id not in self.entities:
            raise ValueError("Both entities must exist")
            
        # Add relationship
        self.relationships[entity1_id].add(entity2_id)
        self.relationships[entity2_id].add(entity1_id)
        
        # Add metadata
        self.relationship_metadata[(entity1_id, entity2_id)] = {
            "type": relationship_type,
            "created_at": datetime.now(),
            **(metadata or {})
        }
        
        # Check limits
        if self.max_relationships:
            total_relationships = sum(len(rels) for rels in self.relationships.values())
            if total_relationships > self.max_relationships:
                await self._prune_relationships()

    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID."""
        if entity_id not in self.entities:
            return None
            
        return {
            **self.entities[entity_id],
            "metadata": self.entity_metadata[entity_id],
            "relationships": list(self.relationships[entity_id])
        }

    async def get_entities_by_type(
        self,
        entity_type: str
    ) -> List[Dict[str, Any]]:
        """Get all entities of a specific type."""
        return [
            {
                **entity,
                "id": entity_id,
                "metadata": self.entity_metadata[entity_id],
                "relationships": list(self.relationships[entity_id])
            }
            for entity_id, entity in self.entities.items()
            if entity["type"] == entity_type
        ]

    async def get_related_entities(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get entities related to the given entity."""
        if entity_id not in self.entities:
            return []
            
        related = []
        for related_id in self.relationships[entity_id]:
            if relationship_type is None or (
                (entity_id, related_id) in self.relationship_metadata and
                self.relationship_metadata[(entity_id, related_id)]["type"] == relationship_type
            ):
                related.append({
                    **self.entities[related_id],
                    "id": related_id,
                    "metadata": self.entity_metadata[related_id],
                    "relationship_metadata": self.relationship_metadata.get((entity_id, related_id))
                })
                
        return related

    async def update_entity(
        self,
        entity_id: str,
        properties: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update entity properties and metadata."""
        if entity_id not in self.entities:
            raise ValueError(f"Entity not found: {entity_id}")
            
        if properties:
            self.entities[entity_id]["properties"].update(properties)
            self.entities[entity_id]["updated_at"] = datetime.now()
            
        if metadata:
            self.entity_metadata[entity_id].update(metadata)

    async def remove_entity(self, entity_id: str) -> None:
        """Remove an entity and its relationships."""
        if entity_id not in self.entities:
            return
            
        # Remove relationships
        for related_id in self.relationships[entity_id]:
            self.relationships[related_id].remove(entity_id)
            if (entity_id, related_id) in self.relationship_metadata:
                del self.relationship_metadata[(entity_id, related_id)]
                
        # Remove entity
        del self.entities[entity_id]
        del self.entity_metadata[entity_id]
        del self.relationships[entity_id]

    async def _prune_entities(self) -> None:
        """Prune entities based on limits."""
        if not self.max_entities:
            return
            
        # Sort by last update
        entities = sorted(
            self.entities.items(),
            key=lambda x: x[1]["updated_at"]
        )
        
        # Remove oldest entities
        while len(self.entities) > self.max_entities:
            entity_id, _ = entities.pop(0)
            await self.remove_entity(entity_id)

    async def _prune_relationships(self) -> None:
        """Prune relationships based on limits."""
        if not self.max_relationships:
            return
            
        # Get all relationships
        all_relationships = []
        for entity_id, related_ids in self.relationships.items():
            for related_id in related_ids:
                if (entity_id, related_id) in self.relationship_metadata:
                    all_relationships.append((
                        (entity_id, related_id),
                        self.relationship_metadata[(entity_id, related_id)]["created_at"]
                    ))
                    
        # Sort by creation time
        all_relationships.sort(key=lambda x: x[1])
        
        # Remove oldest relationships
        while len(all_relationships) > self.max_relationships:
            (entity1_id, entity2_id), _ = all_relationships.pop(0)
            self.relationships[entity1_id].remove(entity2_id)
            self.relationships[entity2_id].remove(entity1_id)
            del self.relationship_metadata[(entity1_id, entity2_id)]

    async def clear(self) -> None:
        """Clear all entities and relationships."""
        self.entities.clear()
        self.relationships.clear()
        self.entity_metadata.clear()
        self.relationship_metadata.clear()

    async def get_entity_count(self) -> int:
        """Get the number of entities."""
        return len(self.entities)

    async def get_relationship_count(self) -> int:
        """Get the number of relationships."""
        return sum(len(rels) for rels in self.relationships.values()) // 2  # Divide by 2 because relationships are bidirectional 