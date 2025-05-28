"""
Knowledge graph memory implementation for storing and querying entity relationships.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
import json
from pathlib import Path
import networkx as nx
from ..models.base import BaseLLM
from .base import BaseMemory

class KnowledgeGraphMemory(BaseMemory):
    """Memory that uses a knowledge graph to store entity relationships."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_entities: int = 1000,
        entity_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_entities = max_entities
        self.entity_types = entity_types or ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "CONCEPT"]
        self.relationship_types = relationship_types or [
            "WORKS_FOR", "LOCATED_IN", "PART_OF", "RELATED_TO", "OCCURRED_AT"
        ]
        self.graph = nx.DiGraph()
        self.messages: List[Dict[str, str]] = []
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message and extract entities and relationships."""
        message_with_timestamp = {
            **message,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message_with_timestamp)
        
        # Extract entities and relationships
        entities, relationships = await self._extract_entities_and_relationships(message["content"])
        
        # Add to graph
        self._update_graph(entities, relationships)
        
        # Trim graph if needed
        if len(self.graph.nodes) > self.max_entities:
            self._trim_graph()
        
        await self.save()

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages."""
        return self.messages

    async def clear(self) -> None:
        """Clear all messages and the graph."""
        self.messages.clear()
        self.graph.clear()
        await self.save()

    async def save(self) -> None:
        """Save messages and graph to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "messages": self.messages,
                    "graph": nx.node_link_data(self.graph)
                }, f)

    def load(self) -> None:
        """Load messages and graph from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.messages = data.get("messages", [])
                graph_data = data.get("graph", {})
                if graph_data:
                    self.graph = nx.node_link_graph(graph_data)

    async def _extract_entities_and_relationships(
        self,
        text: str
    ) -> Tuple[Set[str], List[Tuple[str, str, str]]]:
        """Extract entities and relationships from text using LLM."""
        try:
            # Use LLM to extract entities and relationships with types
            prompt = f"""
            Extract entities and their relationships from the following text.
            For each entity, specify its type from: {', '.join(self.entity_types)}
            For each relationship, specify its type from: {', '.join(self.relationship_types)}
            Format the output as a list of (entity1, entity1_type, relationship, relationship_type, entity2, entity2_type) tuples.
            Text: {text}
            """
            response = await self.llm.generate(prompt)
            
            # Parse response to get entities and relationships
            entities = set()
            relationships = []
            
            # Simple parsing - can be made more robust
            lines = response.strip().split('\n')
            for line in lines:
                if '(' in line and ')' in line:
                    parts = line.strip('()').split(',')
                    if len(parts) == 6:
                        entity1 = parts[0].strip()
                        entity1_type = parts[1].strip()
                        rel = parts[2].strip()
                        rel_type = parts[3].strip()
                        entity2 = parts[4].strip()
                        entity2_type = parts[5].strip()
                        
                        entities.add((entity1, entity1_type))
                        entities.add((entity2, entity2_type))
                        relationships.append((entity1, rel, rel_type, entity2))
            
            return entities, relationships
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return set(), []

    def _update_graph(
        self,
        entities: Set[Tuple[str, str]],
        relationships: List[Tuple[str, str, str, str]]
    ) -> None:
        """Update the knowledge graph with new entities and relationships."""
        # Add entities as nodes with their types
        for entity, entity_type in entities:
            if not self.graph.has_node(entity):
                self.graph.add_node(entity, type=entity_type)
        
        # Add relationships as edges with their types
        for source, rel, rel_type, target in relationships:
            self.graph.add_edge(
                source,
                target,
                relationship=rel,
                type=rel_type,
                timestamp=datetime.now().isoformat()
            )

    def _trim_graph(self) -> None:
        """Trim the graph to maintain max_entities limit."""
        # Remove oldest nodes first
        nodes_by_time = sorted(
            self.graph.nodes(data=True),
            key=lambda x: min(
                edge["timestamp"]
                for edge in self.graph.edges(x[0], data=True)
            ) if self.graph.edges(x[0]) else datetime.now().isoformat()
        )
        
        # Remove oldest nodes until we're under the limit
        while len(self.graph.nodes) > self.max_entities:
            node, _ = nodes_by_time.pop(0)
            self.graph.remove_node(node)

    def get_entity_relationships(self, entity: str) -> List[Dict[str, Any]]:
        """Get all relationships for an entity."""
        if not self.graph.has_node(entity):
            return []
        
        relationships = []
        for _, target, data in self.graph.edges(entity, data=True):
            relationships.append({
                "source": entity,
                "target": target,
                "relationship": data["relationship"],
                "type": data["type"],
                "timestamp": data["timestamp"]
            })
        
        return relationships

    def get_related_entities(self, entity: str, max_depth: int = 2) -> List[str]:
        """Get entities related to the given entity within max_depth."""
        if not self.graph.has_node(entity):
            return []
        
        related = set()
        for node in nx.descendants_at_distance(self.graph, entity, max_depth):
            related.add(node)
        
        return list(related)

    def get_entity_context(self, entity: str) -> str:
        """Get context about an entity from the knowledge graph."""
        if not self.graph.has_node(entity):
            return ""
        
        # Get direct relationships
        relationships = self.get_entity_relationships(entity)
        
        # Format context
        context = [f"Entity: {entity} (Type: {self.graph.nodes[entity]['type']})"]
        for rel in relationships:
            context.append(
                f"{rel['source']} {rel['relationship']} ({rel['type']}) {rel['target']}"
            )
        
        return "\n".join(context)

    def get_entity_types(self) -> Dict[str, int]:
        """Get count of entities by type."""
        type_counts = {}
        for node, data in self.graph.nodes(data=True):
            entity_type = data.get("type", "UNKNOWN")
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        return type_counts

    def get_relationship_types(self) -> Dict[str, int]:
        """Get count of relationships by type."""
        type_counts = {}
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get("type", "UNKNOWN")
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
        return type_counts

    def get_central_entities(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get the most central entities in the graph."""
        if not self.graph.nodes:
            return []
        
        # Calculate centrality metrics
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        pagerank = nx.pagerank(self.graph)
        
        # Combine metrics
        entities = []
        for node in self.graph.nodes():
            entities.append({
                "entity": node,
                "type": self.graph.nodes[node]["type"],
                "degree_centrality": degree_centrality[node],
                "betweenness_centrality": betweenness_centrality[node],
                "pagerank": pagerank[node],
                "score": (
                    degree_centrality[node] +
                    betweenness_centrality[node] +
                    pagerank[node]
                ) / 3
            })
        
        # Sort by combined score
        entities.sort(key=lambda x: x["score"], reverse=True)
        return entities[:top_k]

    def get_entity_clusters(self) -> List[List[str]]:
        """Get clusters of related entities using community detection."""
        if not self.graph.nodes:
            return []
        
        # Convert to undirected graph for community detection
        undirected = self.graph.to_undirected()
        
        # Detect communities
        communities = nx.community.greedy_modularity_communities(undirected)
        
        # Convert to lists of entity names
        return [list(community) for community in communities]

    def get_entity_path(self, source: str, target: str) -> Optional[List[Dict[str, Any]]]:
        """Get the shortest path between two entities."""
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return None
        
        try:
            path = nx.shortest_path(self.graph, source, target)
            result = []
            
            for i in range(len(path) - 1):
                current = path[i]
                next_node = path[i + 1]
                edge_data = self.graph.get_edge_data(current, next_node)
                
                result.append({
                    "from": current,
                    "to": next_node,
                    "relationship": edge_data["relationship"],
                    "type": edge_data["type"]
                })
            
            return result
        except nx.NetworkXNoPath:
            return None 