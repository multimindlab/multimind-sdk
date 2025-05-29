"""
Hierarchical memory implementation that organizes information in a tree structure.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class HierarchicalMemory(BaseMemory):
    """Memory that organizes information in a hierarchical structure."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_depth: int = 5,
        max_children: int = 10,
        similarity_threshold: float = 0.7,
        importance_decay: float = 0.95,
        category_learning_rate: float = 0.1,
        min_category_confidence: float = 0.6,
        evolution_tracking: bool = True,
        semantic_analysis: bool = True,
        node_lifecycle: bool = True
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_depth = max_depth
        self.max_children = max_children
        self.similarity_threshold = similarity_threshold
        self.importance_decay = importance_decay
        self.category_learning_rate = category_learning_rate
        self.min_category_confidence = min_category_confidence
        self.evolution_tracking = evolution_tracking
        self.semantic_analysis = semantic_analysis
        self.node_lifecycle = node_lifecycle
        
        # Initialize tree structure
        self.root = {
            "id": "root",
            "content": "Root",
            "children": [],
            "messages": [],
            "timestamp": datetime.now().isoformat(),
            "importance": 1.0,
            "category_embeddings": [],
            "usage_count": 0,
            "evolution_history": [],
            "semantic_tags": set(),
            "lifecycle_state": "active",
            "creation_time": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        }
        self.node_map: Dict[str, Dict[str, Any]] = {"root": self.root}
        self.category_embeddings: Dict[str, List[float]] = {}
        self.semantic_index: Dict[str, Set[str]] = {}  # tag -> node_ids
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message to the appropriate node in the hierarchy."""
        # Analyze message to determine its category and parent
        category, parent_id, confidence = await self._categorize_message(message)
        
        # Create or get node for this category
        node_id = f"{parent_id}_{category}"
        if node_id not in self.node_map:
            await self._create_node(node_id, category, parent_id)
        
        # Add message to node
        message_with_metadata = {
            **message,
            "timestamp": datetime.now().isoformat(),
            "importance": 1.0
        }
        self.node_map[node_id]["messages"].append(message_with_metadata)
        
        # Update node importance and category embeddings
        await self._update_node_importance(node_id)
        if confidence >= self.min_category_confidence:
            await self._update_category_embeddings(node_id, message["content"])
        
        # Update semantic analysis if enabled
        if self.semantic_analysis:
            await self._update_semantic_analysis(node_id, message["content"])
        
        # Track evolution if enabled
        if self.evolution_tracking:
            await self._track_node_evolution(node_id)
        
        # Update lifecycle if enabled
        if self.node_lifecycle:
            await self._update_node_lifecycle(node_id)
        
        # Maintain hierarchy constraints
        await self._maintain_hierarchy()
        await self.save()

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from the hierarchy."""
        messages = []
        self._collect_messages(self.root, messages)
        return sorted(messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear the entire hierarchy."""
        self.root = {
            "id": "root",
            "content": "Root",
            "children": [],
            "messages": [],
            "timestamp": datetime.now().isoformat()
        }
        self.node_map = {"root": self.root}
        await self.save()

    async def save(self) -> None:
        """Save hierarchy to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "root": self.root,
                    "node_map": self.node_map
                }, f)

    def load(self) -> None:
        """Load hierarchy from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.root = data.get("root", self.root)
                self.node_map = data.get("node_map", self.node_map)

    async def _categorize_message(
        self,
        message: Dict[str, str]
    ) -> tuple[str, str, float]:
        """Categorize message and determine its parent node."""
        try:
            prompt = f"""
            Analyze the following message and determine:
            1. A category that best describes its content
            2. The most appropriate parent category from the existing hierarchy
            3. A confidence score between 0 and 1
            
            Existing categories: {list(self.node_map.keys())}
            Message: {message['content']}
            
            Return the category, parent_id, and confidence score.
            """
            response = await self.llm.generate(prompt)
            
            # Parse response to get category, parent, and confidence
            # For now, use simple defaults
            category = "general"
            parent_id = "root"
            confidence = 0.75
            
            return category, parent_id, confidence
        except Exception as e:
            print(f"Error categorizing message: {e}")
            return "general", "root", 0.75

    async def _create_node(
        self,
        node_id: str,
        category: str,
        parent_id: str
    ) -> None:
        """Create a new node in the hierarchy."""
        if parent_id not in self.node_map:
            raise ValueError(f"Parent node {parent_id} does not exist")
        
        new_node = {
            "id": node_id,
            "content": category,
            "children": [],
            "messages": [],
            "timestamp": datetime.now().isoformat(),
            "importance": 1.0,
            "category_embeddings": [],
            "usage_count": 0,
            "evolution_history": [],
            "semantic_tags": set(),
            "lifecycle_state": "active",
            "creation_time": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        }
        
        self.node_map[node_id] = new_node
        self.node_map[parent_id]["children"].append(node_id)

    async def _maintain_hierarchy(self) -> None:
        """Maintain hierarchy constraints (depth and children limits)."""
        # Check depth
        for node_id, node in self.node_map.items():
            depth = self._get_node_depth(node_id)
            if depth > self.max_depth:
                await self._merge_with_parent(node_id)
        
        # Check children count
        for node_id, node in self.node_map.items():
            if len(node["children"]) > self.max_children:
                await self._merge_similar_children(node_id)

    def _get_node_depth(self, node_id: str) -> int:
        """Get the depth of a node in the hierarchy."""
        depth = 0
        current = node_id
        while current != "root":
            parent = self._get_parent_node(current)
            if not parent:
                break
            current = parent["id"]
            depth += 1
        return depth

    def _get_parent_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get the parent node of a given node."""
        for node in self.node_map.values():
            if node_id in node["children"]:
                return node
        return None

    async def _merge_with_parent(self, node_id: str) -> None:
        """Merge a node with its parent."""
        parent = self._get_parent_node(node_id)
        if not parent:
            return
        
        # Move messages to parent
        parent["messages"].extend(self.node_map[node_id]["messages"])
        
        # Move children to parent
        parent["children"].extend(self.node_map[node_id]["children"])
        
        # Remove node
        parent["children"].remove(node_id)
        del self.node_map[node_id]

    async def _merge_similar_children(self, node_id: str) -> None:
        """Merge similar children of a node."""
        node = self.node_map[node_id]
        children = node["children"]
        
        # Calculate similarities between children
        similarities = {}
        for i, child1 in enumerate(children):
            for child2 in children[i+1:]:
                sim = await self._calculate_similarity(
                    self.node_map[child1]["content"],
                    self.node_map[child2]["content"]
                )
                if sim >= self.similarity_threshold:
                    similarities[(child1, child2)] = sim
        
        # Merge most similar pairs
        for (child1, child2), _ in sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if child1 in node["children"] and child2 in node["children"]:
                await self._merge_nodes(child1, child2)
                if len(node["children"]) <= self.max_children:
                    break

    async def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        try:
            # Get embeddings
            emb1 = await self.llm.embeddings(text1)
            emb2 = await self.llm.embeddings(text2)
            
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = sum(a * a for a in emb1) ** 0.5
            norm2 = sum(b * b for b in emb2) ** 0.5
            
            return dot_product / (norm1 * norm2)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    async def _merge_nodes(self, node1_id: str, node2_id: str) -> None:
        """Merge two nodes."""
        node1 = self.node_map[node1_id]
        node2 = self.node_map[node2_id]
        
        # Create new merged node
        merged_id = f"{node1_id}_{node2_id}"
        merged_node = {
            "id": merged_id,
            "content": f"{node1['content']} + {node2['content']}",
            "children": node1["children"] + node2["children"],
            "messages": node1["messages"] + node2["messages"],
            "timestamp": datetime.now().isoformat(),
            "importance": (node1["importance"] + node2["importance"]) / 2,
            "category_embeddings": (node1["category_embeddings"] + node2["category_embeddings"]) / 2,
            "usage_count": node1["usage_count"] + node2["usage_count"],
            "evolution_history": node1["evolution_history"] + node2["evolution_history"],
            "semantic_tags": node1["semantic_tags"] | node2["semantic_tags"],
            "lifecycle_state": "active",
            "creation_time": node1["creation_time"],
            "last_modified": node1["last_modified"]
        }
        
        # Update parent
        parent = self._get_parent_node(node1_id)
        if parent:
            parent["children"].remove(node1_id)
            parent["children"].remove(node2_id)
            parent["children"].append(merged_id)
        
        # Update node map
        self.node_map[merged_id] = merged_node
        del self.node_map[node1_id]
        del self.node_map[node2_id]

    def _collect_messages(
        self,
        node: Dict[str, Any],
        messages: List[Dict[str, str]]
    ) -> None:
        """Collect messages from a node and its children."""
        messages.extend(node["messages"])
        for child_id in node["children"]:
            if child_id in self.node_map:
                self._collect_messages(self.node_map[child_id], messages)

    async def _update_node_importance(self, node_id: str) -> None:
        """Update node importance based on usage and message importance."""
        node = self.node_map[node_id]
        node["usage_count"] += 1
        
        # Calculate message importance
        message_importance = sum(
            msg.get("importance", 1.0)
            for msg in node["messages"]
        )
        
        # Update node importance
        node["importance"] = (
            self.importance_decay * node["importance"] +
            (1 - self.importance_decay) * message_importance
        )
        
        # Propagate importance to parent
        parent = self._get_parent_node(node_id)
        if parent:
            parent["importance"] = max(
                parent["importance"],
                node["importance"] * 0.8
            )

    async def _update_category_embeddings(
        self,
        node_id: str,
        content: str
    ) -> None:
        """Update category embeddings with new content."""
        try:
            # Get embedding for new content
            new_embedding = await self.llm.embeddings(content)
            
            # Update node's category embeddings
            node = self.node_map[node_id]
            if not node["category_embeddings"]:
                node["category_embeddings"] = [new_embedding]
            else:
                # Update existing embeddings with learning rate
                for i in range(len(node["category_embeddings"])):
                    node["category_embeddings"][i] = [
                        (1 - self.category_learning_rate) * old +
                        self.category_learning_rate * new
                        for old, new in zip(
                            node["category_embeddings"][i],
                            new_embedding
                        )
                    ]
        except Exception as e:
            print(f"Error updating category embeddings: {e}")

    async def query_hierarchy(
        self,
        query: str,
        max_results: int = 5,
        min_importance: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Query the hierarchy for relevant information."""
        try:
            # Get query embedding
            query_embedding = await self.llm.embeddings(query)
            
            # Search through nodes
            results = []
            for node_id, node in self.node_map.items():
                if node["importance"] < min_importance:
                    continue
                
                # Calculate similarity with category embeddings
                max_similarity = 0.0
                for embedding in node["category_embeddings"]:
                    similarity = self._cosine_similarity(
                        query_embedding,
                        embedding
                    )
                    max_similarity = max(max_similarity, similarity)
                
                if max_similarity > 0:
                    results.append({
                        "node_id": node_id,
                        "content": node["content"],
                        "similarity": max_similarity,
                        "importance": node["importance"],
                        "messages": node["messages"]
                    })
            
            # Sort by similarity and importance
            results.sort(
                key=lambda x: x["similarity"] * x["importance"],
                reverse=True
            )
            
            return results[:max_results]
        except Exception as e:
            print(f"Error querying hierarchy: {e}")
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2)

    async def get_important_nodes(
        self,
        min_importance: float = 0.5,
        max_depth: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get nodes with importance above threshold."""
        important_nodes = []
        
        for node_id, node in self.node_map.items():
            if node["importance"] >= min_importance:
                depth = self._get_node_depth(node_id)
                if max_depth is None or depth <= max_depth:
                    important_nodes.append({
                        "node_id": node_id,
                        "content": node["content"],
                        "importance": node["importance"],
                        "depth": depth,
                        "message_count": len(node["messages"]),
                        "usage_count": node["usage_count"]
                    })
        
        return sorted(
            important_nodes,
            key=lambda x: x["importance"],
            reverse=True
        )

    async def get_node_relationships(
        self,
        node_id: str,
        max_distance: int = 2
    ) -> Dict[str, Any]:
        """Get relationships between nodes within a distance."""
        if node_id not in self.node_map:
            return {}
        
        relationships = {
            "node": node_id,
            "content": self.node_map[node_id]["content"],
            "parents": [],
            "children": [],
            "siblings": [],
            "cousins": []
        }
        
        # Get parent
        parent = self._get_parent_node(node_id)
        if parent:
            relationships["parents"].append({
                "node_id": parent["id"],
                "content": parent["content"],
                "importance": parent["importance"]
            })
        
        # Get children
        node = self.node_map[node_id]
        for child_id in node["children"]:
            child = self.node_map[child_id]
            relationships["children"].append({
                "node_id": child_id,
                "content": child["content"],
                "importance": child["importance"]
            })
        
        # Get siblings
        if parent:
            for sibling_id in parent["children"]:
                if sibling_id != node_id:
                    sibling = self.node_map[sibling_id]
                    relationships["siblings"].append({
                        "node_id": sibling_id,
                        "content": sibling["content"],
                        "importance": sibling["importance"]
                    })
        
        # Get cousins (nodes at same depth)
        node_depth = self._get_node_depth(node_id)
        for other_id, other_node in self.node_map.items():
            if other_id != node_id:
                other_depth = self._get_node_depth(other_id)
                if other_depth == node_depth:
                    relationships["cousins"].append({
                        "node_id": other_id,
                        "content": other_node["content"],
                        "importance": other_node["importance"]
                    })
        
        return relationships

    async def get_semantic_analysis(
        self,
        node_id: str
    ) -> Dict[str, Any]:
        """Get semantic analysis for a node."""
        if node_id not in self.node_map:
            return {}
        
        node = self.node_map[node_id]
        return {
            "node_id": node_id,
            "content": node["content"],
            "semantic_tags": list(node["semantic_tags"]),
            "related_nodes": await self._get_related_nodes(node_id),
            "semantic_cluster": await self._get_semantic_cluster(node_id)
        }

    async def _get_related_nodes(
        self,
        node_id: str,
        max_related: int = 5
    ) -> List[Dict[str, Any]]:
        """Get nodes related by semantic tags."""
        if node_id not in self.node_map:
            return []
        
        node = self.node_map[node_id]
        related_nodes = []
        
        # Find nodes sharing semantic tags
        for tag in node["semantic_tags"]:
            if tag in self.semantic_index:
                for related_id in self.semantic_index[tag]:
                    if related_id != node_id:
                        related_node = self.node_map[related_id]
                        related_nodes.append({
                            "node_id": related_id,
                            "content": related_node["content"],
                            "shared_tags": list(
                                node["semantic_tags"] &
                                related_node["semantic_tags"]
                            ),
                            "importance": related_node["importance"]
                        })
        
        # Sort by number of shared tags and importance
        related_nodes.sort(
            key=lambda x: (len(x["shared_tags"]), x["importance"]),
            reverse=True
        )
        
        return related_nodes[:max_related]

    async def _get_semantic_cluster(
        self,
        node_id: str
    ) -> Dict[str, Any]:
        """Get semantic cluster information for a node."""
        if node_id not in self.node_map:
            return {}
        
        node = self.node_map[node_id]
        cluster = {
            "node_id": node_id,
            "content": node["content"],
            "cluster_members": [],
            "cluster_center": None,
            "cluster_density": 0.0
        }
        
        # Get related nodes
        related_nodes = await self._get_related_nodes(node_id, max_related=10)
        
        if related_nodes:
            # Calculate cluster center
            embeddings = [node["category_embeddings"][0]] if node["category_embeddings"] else []
            for related in related_nodes:
                related_node = self.node_map[related["node_id"]]
                if related_node["category_embeddings"]:
                    embeddings.append(related_node["category_embeddings"][0])
            
            if embeddings:
                cluster["cluster_center"] = [
                    sum(emb[i] for emb in embeddings) / len(embeddings)
                    for i in range(len(embeddings[0]))
                ]
            
            # Calculate cluster density
            if cluster["cluster_center"]:
                similarities = []
                for emb in embeddings:
                    similarity = self._cosine_similarity(
                        cluster["cluster_center"],
                        emb
                    )
                    similarities.append(similarity)
                cluster["cluster_density"] = sum(similarities) / len(similarities)
            
            # Add cluster members
            cluster["cluster_members"] = [
                {
                    "node_id": related["node_id"],
                    "content": related["content"],
                    "shared_tags": related["shared_tags"]
                }
                for related in related_nodes
            ]
        
        return cluster

    async def get_node_evolution(
        self,
        node_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get evolution history of a node."""
        if node_id not in self.node_map:
            return []
        
        node = self.node_map[node_id]
        evolution = node["evolution_history"]
        
        if start_time or end_time:
            filtered_evolution = []
            for point in evolution:
                point_time = datetime.fromisoformat(point["timestamp"])
                if start_time and point_time < start_time:
                    continue
                if end_time and point_time > end_time:
                    continue
                filtered_evolution.append(point)
            return filtered_evolution
        
        return evolution

    async def get_lifecycle_stats(self) -> Dict[str, Any]:
        """Get statistics about node lifecycles."""
        stats = {
            "total_nodes": len(self.node_map),
            "lifecycle_states": {
                "active": 0,
                "inactive": 0,
                "archived": 0
            },
            "age_distribution": {
                "new": 0,  # < 1 day
                "young": 0,  # 1-7 days
                "mature": 0,  # 7-30 days
                "old": 0  # > 30 days
            },
            "activity_stats": {
                "high": 0,  # > 10 uses/day
                "medium": 0,  # 1-10 uses/day
                "low": 0  # < 1 use/day
            }
        }
        
        current_time = datetime.now()
        
        for node in self.node_map.values():
            # Count lifecycle states
            stats["lifecycle_states"][node["lifecycle_state"]] += 1
            
            # Calculate age
            creation_time = datetime.fromisoformat(node["creation_time"])
            age_days = (current_time - creation_time).total_seconds() / 86400
            
            if age_days < 1:
                stats["age_distribution"]["new"] += 1
            elif age_days < 7:
                stats["age_distribution"]["young"] += 1
            elif age_days < 30:
                stats["age_distribution"]["mature"] += 1
            else:
                stats["age_distribution"]["old"] += 1
            
            # Calculate activity
            if node["usage_count"] > 0:
                usage_rate = node["usage_count"] / age_days
                if usage_rate > 10:
                    stats["activity_stats"]["high"] += 1
                elif usage_rate > 1:
                    stats["activity_stats"]["medium"] += 1
                else:
                    stats["activity_stats"]["low"] += 1
        
        return stats

    async def get_hierarchy_stats(self) -> Dict[str, Any]:
        """Get statistics about the hierarchy."""
        stats = {
            "total_nodes": len(self.node_map),
            "total_messages": len(self.get_messages()),
            "max_depth": max(self._get_node_depth(node_id) for node_id in self.node_map),
            "node_distribution": {},
            "message_distribution": {},
            "importance_distribution": {
                "high": 0,  # > 0.7
                "medium": 0,  # 0.3-0.7
                "low": 0  # < 0.3
            },
            "category_stats": {},
            "semantic_stats": {
                "total_tags": len(self.semantic_index),
                "tag_distribution": {},
                "avg_tags_per_node": 0.0
            }
        }
        
        # Calculate distributions
        total_tags = 0
        for node_id, node in self.node_map.items():
            depth = self._get_node_depth(node_id)
            stats["node_distribution"][depth] = stats["node_distribution"].get(depth, 0) + 1
            stats["message_distribution"][depth] = stats["message_distribution"].get(depth, 0) + len(node["messages"])
            
            # Importance distribution
            if node["importance"] > 0.7:
                stats["importance_distribution"]["high"] += 1
            elif node["importance"] > 0.3:
                stats["importance_distribution"]["medium"] += 1
            else:
                stats["importance_distribution"]["low"] += 1
            
            # Category stats
            category = node["content"]
            if category not in stats["category_stats"]:
                stats["category_stats"][category] = {
                    "node_count": 0,
                    "message_count": 0,
                    "avg_importance": 0.0
                }
            stats["category_stats"][category]["node_count"] += 1
            stats["category_stats"][category]["message_count"] += len(node["messages"])
            stats["category_stats"][category]["avg_importance"] = (
                (stats["category_stats"][category]["avg_importance"] *
                 (stats["category_stats"][category]["node_count"] - 1) +
                 node["importance"]) /
                stats["category_stats"][category]["node_count"]
            )
            
            # Semantic stats
            total_tags += len(node["semantic_tags"])
            for tag in node["semantic_tags"]:
                stats["semantic_stats"]["tag_distribution"][tag] = \
                    stats["semantic_stats"]["tag_distribution"].get(tag, 0) + 1
        
        # Calculate average tags per node
        if stats["total_nodes"] > 0:
            stats["semantic_stats"]["avg_tags_per_node"] = total_tags / stats["total_nodes"]
        
        return stats

    async def get_node_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for hierarchy optimization."""
        suggestions = []
        
        # Check depth distribution
        depth_dist = {}
        for node_id in self.node_map:
            depth = self._get_node_depth(node_id)
            depth_dist[depth] = depth_dist.get(depth, 0) + 1
        
        # Suggest rebalancing if depth distribution is uneven
        if depth_dist:
            avg_nodes = sum(depth_dist.values()) / len(depth_dist)
            for depth, count in depth_dist.items():
                if count > avg_nodes * 1.5:
                    suggestions.append({
                        "type": "depth_balance",
                        "depth": depth,
                        "suggestion": f"Consider redistributing nodes at depth {depth}"
                    })
        
        # Check children distribution
        for node_id, node in self.node_map.items():
            if len(node["children"]) > self.max_children * 0.8:
                suggestions.append({
                    "type": "children_limit",
                    "node": node_id,
                    "suggestion": f"Consider merging children of node {node_id}"
                })
        
        # Check importance distribution
        for node_id, node in self.node_map.items():
            if node["importance"] < 0.2 and len(node["messages"]) > 10:
                suggestions.append({
                    "type": "low_importance",
                    "node": node_id,
                    "suggestion": f"Consider merging or removing low-importance node {node_id}"
                })
        
        # Check category distribution
        category_counts = {}
        for node in self.node_map.values():
            category = node["content"]
            category_counts[category] = category_counts.get(category, 0) + 1
        
        for category, count in category_counts.items():
            if count > 5:  # Arbitrary threshold
                suggestions.append({
                    "type": "category_consolidation",
                    "category": category,
                    "suggestion": f"Consider consolidating nodes in category '{category}'"
                })
        
        # Check lifecycle states
        lifecycle_stats = await self.get_lifecycle_stats()
        if lifecycle_stats["lifecycle_states"]["archived"] > len(self.node_map) * 0.3:
            suggestions.append({
                "type": "lifecycle_cleanup",
                "suggestion": "Consider cleaning up archived nodes"
            })
        
        # Check semantic tag distribution
        semantic_stats = (await self.get_hierarchy_stats())["semantic_stats"]
        if semantic_stats["avg_tags_per_node"] < 2:
            suggestions.append({
                "type": "semantic_enrichment",
                "suggestion": "Consider enriching nodes with more semantic tags"
            })
        
        return suggestions

    async def _update_semantic_analysis(
        self,
        node_id: str,
        content: str
    ) -> None:
        """Update semantic analysis for a node."""
        try:
            # Extract semantic tags
            prompt = f"""
            Analyze the following content and extract key semantic tags that describe its meaning.
            Return a list of tags separated by commas.
            
            Content: {content}
            """
            response = await self.llm.generate(prompt)
            tags = {tag.strip() for tag in response.split(",")}
            
            # Update node's semantic tags
            node = self.node_map[node_id]
            node["semantic_tags"].update(tags)
            
            # Update semantic index
            for tag in tags:
                if tag not in self.semantic_index:
                    self.semantic_index[tag] = set()
                self.semantic_index[tag].add(node_id)
        except Exception as e:
            print(f"Error updating semantic analysis: {e}")

    async def _track_node_evolution(self, node_id: str) -> None:
        """Track the evolution of a node over time."""
        node = self.node_map[node_id]
        current_state = {
            "timestamp": datetime.now().isoformat(),
            "message_count": len(node["messages"]),
            "importance": node["importance"],
            "children_count": len(node["children"]),
            "semantic_tags": list(node["semantic_tags"]),
            "lifecycle_state": node["lifecycle_state"]
        }
        
        # Add to evolution history
        node["evolution_history"].append(current_state)
        
        # Keep only last 100 evolution points
        if len(node["evolution_history"]) > 100:
            node["evolution_history"] = node["evolution_history"][-100:]

    async def _update_node_lifecycle(self, node_id: str) -> None:
        """Update the lifecycle state of a node."""
        node = self.node_map[node_id]
        current_time = datetime.now()
        last_modified = datetime.fromisoformat(node["last_modified"])
        age = (current_time - datetime.fromisoformat(node["creation_time"])).total_seconds()
        
        # Update lifecycle state based on activity and age
        if node["usage_count"] == 0 and age > 86400:  # 24 hours
            node["lifecycle_state"] = "inactive"
        elif node["importance"] < 0.2 and age > 604800:  # 1 week
            node["lifecycle_state"] = "archived"
        elif node["usage_count"] > 0:
            node["lifecycle_state"] = "active"
        
        # Update last modified time
        node["last_modified"] = current_time.isoformat()

    async def _update_node_lifecycle(self, node_id: str) -> None:
        """Update the lifecycle state of a node."""
        node = self.node_map[node_id]
        current_time = datetime.now()
        last_modified = datetime.fromisoformat(node["last_modified"])
        age = (current_time - datetime.fromisoformat(node["creation_time"])).total_seconds()
        
        # Update lifecycle state based on activity and age
        if node["usage_count"] == 0 and age > 86400:  # 24 hours
            node["lifecycle_state"] = "inactive"
        elif node["importance"] < 0.2 and age > 604800:  # 1 week
            node["lifecycle_state"] = "archived"
        elif node["usage_count"] > 0:
            node["lifecycle_state"] = "active"
        
        # Update last modified time
        node["last_modified"] = current_time.isoformat() 