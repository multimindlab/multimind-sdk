"""
Versioned and snapshot memory implementation.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class VersionedMemory(BaseMemory):
    """Memory that implements versioned/snapshot memory."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_items: int = 1000,
        max_versions: int = 100,
        snapshot_interval: int = 3600,  # 1 hour
        version_retention_days: int = 30,
        enable_differential_snapshots: bool = True,
        differential_threshold: float = 0.3,
        enable_compression: bool = True,
        compression_ratio: float = 0.5,
        enable_metadata_tracking: bool = True,
        metadata_interval: int = 3600,  # 1 hour
        enable_version_analysis: bool = True,
        analysis_interval: int = 3600,  # 1 hour
        enable_optimization: bool = True,
        optimization_interval: int = 3600,  # 1 hour
        enable_branching: bool = True,
        max_branches: int = 10,
        enable_merge_detection: bool = True,
        merge_threshold: float = 0.7,
        enable_conflict_resolution: bool = True,
        conflict_threshold: float = 0.5,
        enable_version_graph: bool = True,
        graph_update_interval: int = 3600  # 1 hour
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_items = max_items
        self.max_versions = max_versions
        self.snapshot_interval = snapshot_interval
        self.version_retention_days = version_retention_days
        self.enable_differential_snapshots = enable_differential_snapshots
        self.differential_threshold = differential_threshold
        self.enable_compression = enable_compression
        self.compression_ratio = compression_ratio
        self.enable_metadata_tracking = enable_metadata_tracking
        self.metadata_interval = metadata_interval
        self.enable_version_analysis = enable_version_analysis
        self.analysis_interval = analysis_interval
        self.enable_optimization = enable_optimization
        self.optimization_interval = optimization_interval
        self.enable_branching = enable_branching
        self.max_branches = max_branches
        self.enable_merge_detection = enable_merge_detection
        self.merge_threshold = merge_threshold
        self.enable_conflict_resolution = enable_conflict_resolution
        self.conflict_threshold = conflict_threshold
        self.enable_version_graph = enable_version_graph
        self.graph_update_interval = graph_update_interval
        
        # Initialize storage
        self.items: List[Dict[str, Any]] = []
        self.versions: Dict[str, List[Dict[str, Any]]] = {}  # item_id -> version history
        self.snapshots: List[Dict[str, Any]] = []  # List of snapshots
        self.differentials: Dict[str, Dict[str, Any]] = {}  # snapshot_id -> differential
        self.metadata: Dict[str, Dict[str, Any]] = {}  # item_id -> metadata
        self.branches: Dict[str, List[str]] = {}  # branch_id -> version chain
        self.merge_points: Dict[str, List[str]] = {}  # merge_id -> merged versions
        self.conflicts: Dict[str, List[Dict[str, Any]]] = {}  # conflict_id -> conflict data
        self.version_graph: Dict[str, Set[str]] = {}  # version_id -> connected versions
        self.last_snapshot = datetime.now()
        self.last_metadata = datetime.now()
        self.last_analysis = datetime.now()
        self.last_optimization = datetime.now()
        self.last_graph_update = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message and create version."""
        # Create new item
        item_id = f"item_{len(self.items)}"
        new_item = {
            "id": item_id,
            "content": message["content"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "type": "message",
                "version": 1,
                "branch": "main",
                "snapshot_id": None,
                "differential_id": None,
                "compression_ratio": 1.0,
                "metadata_version": 1,
                "analysis_version": 1,
                "optimization_version": 1,
                "graph_version": 1
            }
        }
        
        # Add to storage
        self.items.append(new_item)
        
        # Initialize version history
        self.versions[item_id] = [{
            "version": 1,
            "content": message["content"],
            "timestamp": datetime.now().isoformat(),
            "branch": "main",
            "parent": None,
            "metadata": {}
        }]
        
        # Initialize metadata
        if self.enable_metadata_tracking:
            await self._initialize_metadata(item_id)
        
        # Create snapshot if needed
        if (datetime.now() - self.last_snapshot).total_seconds() >= self.snapshot_interval:
            await self._create_snapshot()
        
        # Create differential if enabled
        if self.enable_differential_snapshots:
            await self._create_differential(item_id)
        
        # Compress if enabled
        if self.enable_compression:
            await self._compress_version(item_id)
        
        # Analyze version if enabled
        if self.enable_version_analysis:
            await self._analyze_version(item_id)
        
        # Update version graph if enabled
        if self.enable_version_graph:
            await self._update_version_graph(item_id)
        
        # Check for merges if enabled
        if self.enable_merge_detection:
            await self._detect_merges(item_id)
        
        # Check for conflicts if enabled
        if self.enable_conflict_resolution:
            await self._detect_conflicts(item_id)
        
        # Maintain item limit
        await self._maintain_item_limit()
        
        await self.save()

    async def _initialize_metadata(self, item_id: str) -> None:
        """Initialize metadata for an item."""
        item = next(i for i in self.items if i["id"] == item_id)
        
        try:
            # Generate metadata prompt
            prompt = f"""
            Generate metadata for this item:
            
            {item['content']}
            
            Return a JSON object with:
            1. metadata: dict of string -> any
            2. metadata_version: int
            3. metadata_reason: string
            """
            response = await self.llm.generate(prompt)
            metadata = json.loads(response)
            
            # Update item metadata
            self.metadata[item_id] = metadata["metadata"]
            item["metadata"]["metadata_version"] = metadata["metadata_version"]
            
        except Exception as e:
            print(f"Error initializing metadata: {e}")

    async def _create_snapshot(self) -> None:
        """Create a snapshot of current state."""
        try:
            snapshot = {
                "id": f"snapshot_{len(self.snapshots)}",
                "timestamp": datetime.now().isoformat(),
                "items": [
                    {
                        "id": item["id"],
                        "content": item["content"],
                        "version": item["metadata"]["version"],
                        "branch": item["metadata"]["branch"]
                    }
                    for item in self.items
                ],
                "metadata": {
                    "total_items": len(self.items),
                    "total_versions": sum(len(v) for v in self.versions.values()),
                    "total_branches": len(self.branches),
                    "total_merges": len(self.merge_points),
                    "total_conflicts": len(self.conflicts)
                }
            }
            
            self.snapshots.append(snapshot)
            self.last_snapshot = datetime.now()
            
        except Exception as e:
            print(f"Error creating snapshot: {e}")

    async def _create_differential(self, item_id: str) -> None:
        """Create differential for an item."""
        item = next(i for i in self.items if i["id"] == item_id)
        version_history = self.versions[item_id]
        
        if len(version_history) > 1:
            try:
                # Calculate differential
                current_version = version_history[-1]
                previous_version = version_history[-2]
                
                differential = {
                    "id": f"diff_{item_id}_{current_version['version']}",
                    "item_id": item_id,
                    "from_version": previous_version["version"],
                    "to_version": current_version["version"],
                    "timestamp": datetime.now().isoformat(),
                    "changes": {
                        "content_changes": self._calculate_content_changes(
                            previous_version["content"],
                            current_version["content"]
                        ),
                        "metadata_changes": self._calculate_metadata_changes(
                            previous_version.get("metadata", {}),
                            current_version.get("metadata", {})
                        )
                    }
                }
                
                self.differentials[differential["id"]] = differential
                item["metadata"]["differential_id"] = differential["id"]
                
            except Exception as e:
                print(f"Error creating differential: {e}")

    def _calculate_content_changes(self, old_content: str, new_content: str) -> Dict[str, Any]:
        """Calculate changes between content versions."""
        # This is a simplified version - in practice, you'd want to use a proper diff algorithm
        return {
            "added": len(new_content) - len(old_content),
            "changed": sum(1 for a, b in zip(old_content, new_content) if a != b),
            "deleted": len(old_content) - len(new_content)
        }

    def _calculate_metadata_changes(self, old_metadata: Dict[str, Any], new_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate changes between metadata versions."""
        changes = {
            "added": {},
            "modified": {},
            "deleted": {}
        }
        
        # Find added and modified fields
        for key, value in new_metadata.items():
            if key not in old_metadata:
                changes["added"][key] = value
            elif old_metadata[key] != value:
                changes["modified"][key] = {
                    "old": old_metadata[key],
                    "new": value
                }
        
        # Find deleted fields
        for key in old_metadata:
            if key not in new_metadata:
                changes["deleted"][key] = old_metadata[key]
        
        return changes

    async def _compress_version(self, item_id: str) -> None:
        """Compress version history for an item."""
        item = next(i for i in self.items if i["id"] == item_id)
        version_history = self.versions[item_id]
        
        if len(version_history) > 1:
            try:
                # Calculate compression ratio
                original_size = sum(len(v["content"]) for v in version_history)
                compressed_size = len(version_history[-1]["content"])
                ratio = compressed_size / original_size
                
                # Update compression ratio
                item["metadata"]["compression_ratio"] = ratio
                
                # If ratio is below threshold, compress
                if ratio < self.compression_ratio:
                    # Keep only the latest version and its differential
                    latest_version = version_history[-1]
                    version_history.clear()
                    version_history.append(latest_version)
                    
            except Exception as e:
                print(f"Error compressing version: {e}")

    async def _analyze_version(self, item_id: str) -> None:
        """Analyze version history for an item."""
        item = next(i for i in self.items if i["id"] == item_id)
        version_history = self.versions[item_id]
        
        try:
            # Generate version analysis prompt
            prompt = f"""
            Analyze version history for this item:
            
            {item['content']}
            
            Version history:
            {json.dumps(version_history, indent=2)}
            
            Return a JSON object with:
            1. analysis: dict of string -> any
            2. analysis_version: int
            3. analysis_reason: string
            """
            response = await self.llm.generate(prompt)
            analysis = json.loads(response)
            
            # Update item metadata
            item["metadata"]["analysis_version"] = analysis["analysis_version"]
            
        except Exception as e:
            print(f"Error analyzing version: {e}")

    async def _update_version_graph(self, item_id: str) -> None:
        """Update version graph for an item."""
        item = next(i for i in self.items if i["id"] == item_id)
        version_history = self.versions[item_id]
        
        try:
            # Add version to graph
            current_version = version_history[-1]
            version_id = f"{item_id}_v{current_version['version']}"
            
            # Initialize version node if not exists
            if version_id not in self.version_graph:
                self.version_graph[version_id] = set()
            
            # Add edges to parent versions
            if current_version["parent"]:
                parent_id = f"{item_id}_v{current_version['parent']}"
                self.version_graph[version_id].add(parent_id)
                self.version_graph[parent_id].add(version_id)
            
            # Update last graph update time
            self.last_graph_update = datetime.now()
            
        except Exception as e:
            print(f"Error updating version graph: {e}")

    async def _detect_merges(self, item_id: str) -> None:
        """Detect potential merges for an item."""
        item = next(i for i in self.items if i["id"] == item_id)
        version_history = self.versions[item_id]
        
        try:
            # Check for parallel versions
            parallel_versions = [
                v for v in version_history
                if v["version"] > 1 and v["parent"] == version_history[-2]["version"]
            ]
            
            if len(parallel_versions) > 1:
                # Calculate similarity between parallel versions
                similarities = []
                for v1 in parallel_versions:
                    for v2 in parallel_versions:
                        if v1["version"] < v2["version"]:
                            similarity = self._calculate_similarity(
                                v1["content"],
                                v2["content"]
                            )
                            similarities.append({
                                "v1": v1["version"],
                                "v2": v2["version"],
                                "similarity": similarity
                            })
                
                # Check for potential merges
                for sim in similarities:
                    if sim["similarity"] > self.merge_threshold:
                        merge_id = f"merge_{item_id}_{sim['v1']}_{sim['v2']}"
                        self.merge_points[merge_id] = [sim["v1"], sim["v2"]]
            
        except Exception as e:
            print(f"Error detecting merges: {e}")

    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content versions."""
        # This is a simplified version - in practice, you'd want to use a proper similarity metric
        words1 = set(content1.split())
        words2 = set(content2.split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0

    async def _detect_conflicts(self, item_id: str) -> None:
        """Detect potential conflicts for an item."""
        item = next(i for i in self.items if i["id"] == item_id)
        version_history = self.versions[item_id]
        
        try:
            # Check for conflicting changes
            if len(version_history) > 1:
                current_version = version_history[-1]
                previous_version = version_history[-2]
                
                # Calculate conflict score
                conflict_score = self._calculate_conflict_score(
                    previous_version["content"],
                    current_version["content"]
                )
                
                if conflict_score > self.conflict_threshold:
                    conflict_id = f"conflict_{item_id}_{current_version['version']}"
                    self.conflicts[conflict_id] = [{
                        "version": current_version["version"],
                        "content": current_version["content"],
                        "conflict_score": conflict_score,
                        "conflict_type": "content_conflict"
                    }]
            
        except Exception as e:
            print(f"Error detecting conflicts: {e}")

    def _calculate_conflict_score(self, content1: str, content2: str) -> float:
        """Calculate conflict score between two content versions."""
        # This is a simplified version - in practice, you'd want to use a proper conflict detection algorithm
        changes1 = set(content1.split())
        changes2 = set(content2.split())
        intersection = changes1.intersection(changes2)
        union = changes1.union(changes2)
        return 1.0 - (len(intersection) / len(union)) if union else 0.0

    async def _maintain_item_limit(self) -> None:
        """Maintain item limit by removing oldest versions."""
        if len(self.items) > self.max_items:
            # Sort items by timestamp
            sorted_items = sorted(
                self.items,
                key=lambda x: datetime.fromisoformat(x["timestamp"])
            )
            
            # Remove oldest items
            items_to_remove = sorted_items[:len(self.items) - self.max_items]
            for item in items_to_remove:
                await self._remove_item(item["id"])

    async def _remove_item(self, item_id: str) -> None:
        """Remove an item and its associated data."""
        # Remove from items
        self.items = [i for i in self.items if i["id"] != item_id]
        
        # Remove from versions
        if item_id in self.versions:
            del self.versions[item_id]
        
        # Remove from metadata
        if item_id in self.metadata:
            del self.metadata[item_id]
        
        # Remove from differentials
        differentials_to_remove = [
            diff_id for diff_id, diff in self.differentials.items()
            if diff["item_id"] == item_id
        ]
        for diff_id in differentials_to_remove:
            del self.differentials[diff_id]
        
        # Remove from branches
        branches_to_remove = [
            branch_id for branch_id, versions in self.branches.items()
            if any(v.startswith(item_id) for v in versions)
        ]
        for branch_id in branches_to_remove:
            del self.branches[branch_id]
        
        # Remove from merge points
        merges_to_remove = [
            merge_id for merge_id, versions in self.merge_points.items()
            if any(v.startswith(item_id) for v in versions)
        ]
        for merge_id in merges_to_remove:
            del self.merge_points[merge_id]
        
        # Remove from conflicts
        conflicts_to_remove = [
            conflict_id for conflict_id, conflicts in self.conflicts.items()
            if any(c["version"].startswith(item_id) for c in conflicts)
        ]
        for conflict_id in conflicts_to_remove:
            del self.conflicts[conflict_id]
        
        # Remove from version graph
        versions_to_remove = [
            version_id for version_id in self.version_graph
            if version_id.startswith(item_id)
        ]
        for version_id in versions_to_remove:
            del self.version_graph[version_id]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from all items."""
        messages = []
        for item in self.items:
            messages.append({
                "role": "versioned_memory",
                "content": item["content"],
                "timestamp": item["timestamp"]
            })
        return sorted(messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear all items."""
        self.items = []
        self.versions = {}
        self.snapshots = []
        self.differentials = {}
        self.metadata = {}
        self.branches = {}
        self.merge_points = {}
        self.conflicts = {}
        self.version_graph = {}
        await self.save()

    async def save(self) -> None:
        """Save items to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "items": self.items,
                    "versions": self.versions,
                    "snapshots": self.snapshots,
                    "differentials": self.differentials,
                    "metadata": self.metadata,
                    "branches": self.branches,
                    "merge_points": self.merge_points,
                    "conflicts": self.conflicts,
                    "version_graph": {
                        k: list(v) for k, v in self.version_graph.items()
                    },
                    "last_snapshot": self.last_snapshot.isoformat(),
                    "last_metadata": self.last_metadata.isoformat(),
                    "last_analysis": self.last_analysis.isoformat(),
                    "last_optimization": self.last_optimization.isoformat(),
                    "last_graph_update": self.last_graph_update.isoformat()
                }, f)

    def load(self) -> None:
        """Load items from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.items = data.get("items", [])
                self.versions = data.get("versions", {})
                self.snapshots = data.get("snapshots", [])
                self.differentials = data.get("differentials", {})
                self.metadata = data.get("metadata", {})
                self.branches = data.get("branches", {})
                self.merge_points = data.get("merge_points", {})
                self.conflicts = data.get("conflicts", {})
                self.version_graph = {
                    k: set(v) for k, v in data.get("version_graph", {}).items()
                }
                self.last_snapshot = datetime.fromisoformat(
                    data.get("last_snapshot", datetime.now().isoformat())
                )
                self.last_metadata = datetime.fromisoformat(
                    data.get("last_metadata", datetime.now().isoformat())
                )
                self.last_analysis = datetime.fromisoformat(
                    data.get("last_analysis", datetime.now().isoformat())
                )
                self.last_optimization = datetime.fromisoformat(
                    data.get("last_optimization", datetime.now().isoformat())
                )
                self.last_graph_update = datetime.fromisoformat(
                    data.get("last_graph_update", datetime.now().isoformat())
                )

    async def get_versioned_stats(self) -> Dict[str, Any]:
        """Get statistics about versioned memory."""
        stats = {
            "total_items": len(self.items),
            "version_stats": {
                "total_versions": sum(len(v) for v in self.versions.values()),
                "average_versions": sum(len(v) for v in self.versions.values()) / len(self.versions) if self.versions else 0,
                "max_versions": max(len(v) for v in self.versions.values()) if self.versions else 0
            },
            "snapshot_stats": {
                "total_snapshots": len(self.snapshots),
                "latest_snapshot": self.snapshots[-1]["timestamp"] if self.snapshots else None,
                "snapshot_frequency": self.snapshot_interval
            },
            "differential_stats": {
                "total_differentials": len(self.differentials),
                "average_changes": sum(
                    len(diff["changes"]["content_changes"])
                    for diff in self.differentials.values()
                ) / len(self.differentials) if self.differentials else 0
            },
            "branch_stats": {
                "total_branches": len(self.branches),
                "average_branch_length": sum(
                    len(versions) for versions in self.branches.values()
                ) / len(self.branches) if self.branches else 0
            },
            "merge_stats": {
                "total_merges": len(self.merge_points),
                "merge_frequency": len(self.merge_points) / len(self.items) if self.items else 0
            },
            "conflict_stats": {
                "total_conflicts": len(self.conflicts),
                "conflict_frequency": len(self.conflicts) / len(self.items) if self.items else 0
            },
            "graph_stats": {
                "total_nodes": len(self.version_graph),
                "total_edges": sum(len(edges) for edges in self.version_graph.values()),
                "average_degree": sum(len(edges) for edges in self.version_graph.values()) / len(self.version_graph) if self.version_graph else 0
            }
        }
        
        return stats

    async def get_versioned_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for versioned memory optimization."""
        suggestions = []
        
        # Check item count
        if len(self.items) > self.max_items * 0.8:
            suggestions.append({
                "type": "item_limit",
                "suggestion": "Consider increasing max_items or removing older versions"
            })
        
        # Check version count
        stats = await self.get_versioned_stats()
        if stats["version_stats"]["average_versions"] > self.max_versions * 0.8:
            suggestions.append({
                "type": "version_limit",
                "suggestion": "Consider increasing max_versions or compressing version history"
            })
        
        # Check snapshot frequency
        if len(self.snapshots) < 2:
            suggestions.append({
                "type": "snapshot_frequency",
                "suggestion": "Consider adjusting snapshot interval"
            })
        
        # Check differential coverage
        if stats["differential_stats"]["total_differentials"] < len(self.items) * 0.8:
            suggestions.append({
                "type": "differential_coverage",
                "suggestion": "Consider improving differential creation"
            })
        
        # Check branch management
        if stats["branch_stats"]["total_branches"] > self.max_branches * 0.8:
            suggestions.append({
                "type": "branch_limit",
                "suggestion": "Consider increasing max_branches or merging branches"
            })
        
        # Check merge frequency
        if stats["merge_stats"]["merge_frequency"] > 0.5:
            suggestions.append({
                "type": "merge_frequency",
                "suggestion": "Consider adjusting merge detection threshold"
            })
        
        # Check conflict frequency
        if stats["conflict_stats"]["conflict_frequency"] > 0.3:
            suggestions.append({
                "type": "conflict_frequency",
                "suggestion": "Consider adjusting conflict detection threshold"
            })
        
        # Check graph complexity
        if stats["graph_stats"]["average_degree"] > 5:
            suggestions.append({
                "type": "graph_complexity",
                "suggestion": "Consider simplifying version graph"
            })
        
        return suggestions 