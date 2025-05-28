"""
Novelty and salience filtering memory implementation.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class NoveltyMemory(BaseMemory):
    """Memory that implements novelty and salience filtering."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_items: int = 1000,
        novelty_threshold: float = 0.3,
        salience_threshold: float = 0.5,
        novelty_decay_rate: float = 0.1,
        salience_decay_rate: float = 0.05,
        enable_semantic_novelty: bool = True,
        semantic_interval: int = 3600,  # 1 hour
        enable_context_novelty: bool = True,
        context_window: int = 10,
        enable_pattern_novelty: bool = True,
        pattern_threshold: float = 0.7,
        enable_salience_analysis: bool = True,
        salience_interval: int = 3600,  # 1 hour
        enable_importance_weighting: bool = True,
        importance_threshold: float = 0.7,
        enable_optimization: bool = True,
        optimization_interval: int = 3600,  # 1 hour
        enable_temporal_novelty: bool = True,
        temporal_window: int = 24,  # hours
        enable_concept_novelty: bool = True,
        concept_threshold: float = 0.6,
        enable_relation_novelty: bool = True,
        relation_threshold: float = 0.5,
        enable_adaptive_thresholds: bool = True,
        adaptation_interval: int = 3600  # 1 hour
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_items = max_items
        self.novelty_threshold = novelty_threshold
        self.salience_threshold = salience_threshold
        self.novelty_decay_rate = novelty_decay_rate
        self.salience_decay_rate = salience_decay_rate
        self.enable_semantic_novelty = enable_semantic_novelty
        self.semantic_interval = semantic_interval
        self.enable_context_novelty = enable_context_novelty
        self.context_window = context_window
        self.enable_pattern_novelty = enable_pattern_novelty
        self.pattern_threshold = pattern_threshold
        self.enable_salience_analysis = enable_salience_analysis
        self.salience_interval = salience_interval
        self.enable_importance_weighting = enable_importance_weighting
        self.importance_threshold = importance_threshold
        self.enable_optimization = enable_optimization
        self.optimization_interval = optimization_interval
        self.enable_temporal_novelty = enable_temporal_novelty
        self.temporal_window = temporal_window
        self.enable_concept_novelty = enable_concept_novelty
        self.concept_threshold = concept_threshold
        self.enable_relation_novelty = enable_relation_novelty
        self.relation_threshold = relation_threshold
        self.enable_adaptive_thresholds = enable_adaptive_thresholds
        self.adaptation_interval = adaptation_interval
        
        # Initialize storage
        self.items: List[Dict[str, Any]] = []
        self.novelty_scores: Dict[str, float] = {}  # item_id -> novelty score
        self.salience_scores: Dict[str, float] = {}  # item_id -> salience score
        self.semantic_vectors: Dict[str, List[float]] = {}  # item_id -> semantic vector
        self.pattern_matches: Dict[str, Set[str]] = {}  # item_id -> matching patterns
        self.concept_maps: Dict[str, Dict[str, float]] = {}  # item_id -> concept scores
        self.relation_graphs: Dict[str, Dict[str, float]] = {}  # item_id -> relation scores
        self.temporal_windows: Dict[str, List[Dict[str, Any]]] = {}  # item_id -> temporal context
        self.last_semantic = datetime.now()
        self.last_salience = datetime.now()
        self.last_optimization = datetime.now()
        self.last_adaptation = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message and calculate novelty/salience scores."""
        # Create new item
        item_id = f"item_{len(self.items)}"
        new_item = {
            "id": item_id,
            "content": message["content"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "type": "message",
                "novelty_score": 0.0,
                "salience_score": 0.0,
                "importance_score": 0.0,
                "semantic_novelty": 0.0,
                "context_novelty": 0.0,
                "pattern_novelty": 0.0,
                "temporal_novelty": 0.0,
                "concept_novelty": 0.0,
                "relation_novelty": 0.0,
                "semantic_vector": [],
                "patterns": [],
                "context": [],
                "concepts": {},
                "relations": {}
            }
        }
        
        # Add to storage
        self.items.append(new_item)
        
        # Calculate initial scores
        await self._calculate_novelty(item_id)
        await self._calculate_salience(item_id)
        
        # Calculate semantic vector
        if self.enable_semantic_novelty:
            await self._calculate_semantic_vector(item_id)
        
        # Analyze patterns
        if self.enable_pattern_novelty:
            await self._analyze_patterns(item_id)
        
        # Update context
        if self.enable_context_novelty:
            await self._update_context(item_id)
        
        # Analyze temporal novelty
        if self.enable_temporal_novelty:
            await self._analyze_temporal_novelty(item_id)
        
        # Analyze concept novelty
        if self.enable_concept_novelty:
            await self._analyze_concept_novelty(item_id)
        
        # Analyze relation novelty
        if self.enable_relation_novelty:
            await self._analyze_relation_novelty(item_id)
        
        # Adapt thresholds if needed
        if self.enable_adaptive_thresholds and (
            datetime.now() - self.last_adaptation
        ).total_seconds() >= self.adaptation_interval:
            await self._adapt_thresholds()
        
        # Maintain item limit
        await self._maintain_item_limit()
        
        await self.save()

    async def _calculate_novelty(self, item_id: str) -> None:
        """Calculate novelty score for an item."""
        item = next(i for i in self.items if i["id"] == item_id)
        
        try:
            # Generate novelty analysis prompt
            prompt = f"""
            Analyze the novelty of this item:
            
            {item['content']}
            
            Return a JSON object with:
            1. novelty_score: float (0-1)
            2. novelty_factors: list of strings
            3. novelty_reason: string
            """
            response = await self.llm.generate(prompt)
            novelty = json.loads(response)
            
            # Update item metadata
            item["metadata"]["novelty_score"] = novelty["novelty_score"]
            self.novelty_scores[item_id] = novelty["novelty_score"]
            
        except Exception as e:
            print(f"Error calculating novelty: {e}")

    async def _calculate_salience(self, item_id: str) -> None:
        """Calculate salience score for an item."""
        item = next(i for i in self.items if i["id"] == item_id)
        
        try:
            # Generate salience analysis prompt
            prompt = f"""
            Analyze the salience of this item:
            
            {item['content']}
            
            Return a JSON object with:
            1. salience_score: float (0-1)
            2. salience_factors: list of strings
            3. salience_reason: string
            """
            response = await self.llm.generate(prompt)
            salience = json.loads(response)
            
            # Update item metadata
            item["metadata"]["salience_score"] = salience["salience_score"]
            self.salience_scores[item_id] = salience["salience_score"]
            
        except Exception as e:
            print(f"Error calculating salience: {e}")

    async def _calculate_semantic_vector(self, item_id: str) -> None:
        """Calculate semantic vector for an item."""
        item = next(i for i in self.items if i["id"] == item_id)
        
        try:
            # Generate semantic vector prompt
            prompt = f"""
            Generate a semantic vector for this item:
            
            {item['content']}
            
            Return a JSON object with:
            1. semantic_vector: list of floats
            2. vector_dimensions: list of strings
            """
            response = await self.llm.generate(prompt)
            semantic = json.loads(response)
            
            # Update item metadata
            item["metadata"]["semantic_vector"] = semantic["semantic_vector"]
            self.semantic_vectors[item_id] = semantic["semantic_vector"]
            
        except Exception as e:
            print(f"Error calculating semantic vector: {e}")

    async def _analyze_patterns(self, item_id: str) -> None:
        """Analyze patterns in an item."""
        item = next(i for i in self.items if i["id"] == item_id)
        
        try:
            # Generate pattern analysis prompt
            prompt = f"""
            Analyze patterns in this item:
            
            {item['content']}
            
            Return a JSON object with:
            1. patterns: list of strings
            2. pattern_types: list of strings
            3. pattern_confidence: list of floats
            """
            response = await self.llm.generate(prompt)
            patterns = json.loads(response)
            
            # Update item metadata
            item["metadata"]["patterns"] = patterns["patterns"]
            self.pattern_matches[item_id] = set(patterns["patterns"])
            
        except Exception as e:
            print(f"Error analyzing patterns: {e}")

    async def _update_context(self, item_id: str) -> None:
        """Update context for an item."""
        item = next(i for i in self.items if i["id"] == item_id)
        
        # Get recent items
        recent_items = self.items[-self.context_window:]
        
        # Update context
        item["metadata"]["context"] = [
            {
                "id": i["id"],
                "content": i["content"],
                "timestamp": i["timestamp"]
            }
            for i in recent_items
            if i["id"] != item_id
        ]

    async def _analyze_temporal_novelty(self, item_id: str) -> None:
        """Analyze temporal novelty of an item."""
        item = next(i for i in self.items if i["id"] == item_id)
        
        try:
            # Generate temporal analysis prompt
            prompt = f"""
            Analyze temporal novelty of this item:
            
            {item['content']}
            
            Return a JSON object with:
            1. temporal_novelty: float (0-1)
            2. temporal_factors: list of strings
            3. temporal_reason: string
            """
            response = await self.llm.generate(prompt)
            temporal = json.loads(response)
            
            # Update item metadata
            item["metadata"]["temporal_novelty"] = temporal["temporal_novelty"]
            
            # Update temporal window
            self.temporal_windows[item_id] = [
                {
                    "timestamp": item["timestamp"],
                    "novelty": temporal["temporal_novelty"],
                    "factors": temporal["temporal_factors"]
                }
            ]
            
        except Exception as e:
            print(f"Error analyzing temporal novelty: {e}")

    async def _analyze_concept_novelty(self, item_id: str) -> None:
        """Analyze concept novelty of an item."""
        item = next(i for i in self.items if i["id"] == item_id)
        
        try:
            # Generate concept analysis prompt
            prompt = f"""
            Analyze concept novelty of this item:
            
            {item['content']}
            
            Return a JSON object with:
            1. concepts: dict of string -> float (concept -> novelty score)
            2. concept_types: list of strings
            3. concept_reason: string
            """
            response = await self.llm.generate(prompt)
            concepts = json.loads(response)
            
            # Update item metadata
            item["metadata"]["concepts"] = concepts["concepts"]
            self.concept_maps[item_id] = concepts["concepts"]
            
            # Calculate overall concept novelty
            concept_novelty = sum(concepts["concepts"].values()) / len(concepts["concepts"])
            item["metadata"]["concept_novelty"] = concept_novelty
            
        except Exception as e:
            print(f"Error analyzing concept novelty: {e}")

    async def _analyze_relation_novelty(self, item_id: str) -> None:
        """Analyze relation novelty of an item."""
        item = next(i for i in self.items if i["id"] == item_id)
        
        try:
            # Generate relation analysis prompt
            prompt = f"""
            Analyze relation novelty of this item:
            
            {item['content']}
            
            Return a JSON object with:
            1. relations: dict of string -> float (relation -> novelty score)
            2. relation_types: list of strings
            3. relation_reason: string
            """
            response = await self.llm.generate(prompt)
            relations = json.loads(response)
            
            # Update item metadata
            item["metadata"]["relations"] = relations["relations"]
            self.relation_graphs[item_id] = relations["relations"]
            
            # Calculate overall relation novelty
            relation_novelty = sum(relations["relations"].values()) / len(relations["relations"])
            item["metadata"]["relation_novelty"] = relation_novelty
            
        except Exception as e:
            print(f"Error analyzing relation novelty: {e}")

    async def _adapt_thresholds(self) -> None:
        """Adapt novelty and salience thresholds based on recent items."""
        try:
            # Calculate average scores
            avg_novelty = sum(self.novelty_scores.values()) / len(self.novelty_scores)
            avg_salience = sum(self.salience_scores.values()) / len(self.salience_scores)
            
            # Adjust thresholds
            self.novelty_threshold = max(0.1, min(0.9, avg_novelty * 0.8))
            self.salience_threshold = max(0.1, min(0.9, avg_salience * 0.8))
            
            # Update last adaptation time
            self.last_adaptation = datetime.now()
            
        except Exception as e:
            print(f"Error adapting thresholds: {e}")

    async def _update_scores(self, item_id: str) -> None:
        """Update novelty and salience scores over time."""
        item = next(i for i in self.items if i["id"] == item_id)
        
        # Calculate time since last update
        last_update = datetime.fromisoformat(item["timestamp"])
        time_diff = (datetime.now() - last_update).total_seconds()
        
        # Update novelty score
        current_novelty = self.novelty_scores[item_id]
        novelty_decay = np.exp(-self.novelty_decay_rate * time_diff)
        new_novelty = current_novelty * novelty_decay
        self.novelty_scores[item_id] = new_novelty
        item["metadata"]["novelty_score"] = new_novelty
        
        # Update salience score
        current_salience = self.salience_scores[item_id]
        salience_decay = np.exp(-self.salience_decay_rate * time_diff)
        new_salience = current_salience * salience_decay
        self.salience_scores[item_id] = new_salience
        item["metadata"]["salience_score"] = new_salience

    async def _maintain_item_limit(self) -> None:
        """Maintain item limit by removing least novel/salient items."""
        if len(self.items) > self.max_items:
            # Calculate combined scores
            scores = {
                item["id"]: (
                    self.novelty_scores[item["id"]] * 0.4 +
                    self.salience_scores[item["id"]] * 0.6
                )
                for item in self.items
            }
            
            # Sort items by combined score
            sorted_items = sorted(
                self.items,
                key=lambda x: scores[x["id"]]
            )
            
            # Remove lowest scoring items
            items_to_remove = sorted_items[:len(self.items) - self.max_items]
            for item in items_to_remove:
                await self._remove_item(item["id"])

    async def _remove_item(self, item_id: str) -> None:
        """Remove an item and its associated data."""
        # Remove from items
        self.items = [i for i in self.items if i["id"] != item_id]
        
        # Remove from scores
        if item_id in self.novelty_scores:
            del self.novelty_scores[item_id]
        if item_id in self.salience_scores:
            del self.salience_scores[item_id]
        
        # Remove from semantic vectors
        if item_id in self.semantic_vectors:
            del self.semantic_vectors[item_id]
        
        # Remove from pattern matches
        if item_id in self.pattern_matches:
            del self.pattern_matches[item_id]
        
        # Remove from concept maps
        if item_id in self.concept_maps:
            del self.concept_maps[item_id]
        
        # Remove from relation graphs
        if item_id in self.relation_graphs:
            del self.relation_graphs[item_id]
        
        # Remove from temporal windows
        if item_id in self.temporal_windows:
            del self.temporal_windows[item_id]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from all items."""
        messages = []
        for item in self.items:
            messages.append({
                "role": "novelty_memory",
                "content": item["content"],
                "timestamp": item["timestamp"]
            })
        return sorted(messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear all items."""
        self.items = []
        self.novelty_scores = {}
        self.salience_scores = {}
        self.semantic_vectors = {}
        self.pattern_matches = {}
        self.concept_maps = {}
        self.relation_graphs = {}
        self.temporal_windows = {}
        await self.save()

    async def save(self) -> None:
        """Save items to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "items": self.items,
                    "novelty_scores": self.novelty_scores,
                    "salience_scores": self.salience_scores,
                    "semantic_vectors": self.semantic_vectors,
                    "pattern_matches": {
                        k: list(v) for k, v in self.pattern_matches.items()
                    },
                    "concept_maps": self.concept_maps,
                    "relation_graphs": self.relation_graphs,
                    "temporal_windows": self.temporal_windows,
                    "last_semantic": self.last_semantic.isoformat(),
                    "last_salience": self.last_salience.isoformat(),
                    "last_optimization": self.last_optimization.isoformat(),
                    "last_adaptation": self.last_adaptation.isoformat()
                }, f)

    def load(self) -> None:
        """Load items from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.items = data.get("items", [])
                self.novelty_scores = data.get("novelty_scores", {})
                self.salience_scores = data.get("salience_scores", {})
                self.semantic_vectors = data.get("semantic_vectors", {})
                self.pattern_matches = {
                    k: set(v) for k, v in data.get("pattern_matches", {}).items()
                }
                self.concept_maps = data.get("concept_maps", {})
                self.relation_graphs = data.get("relation_graphs", {})
                self.temporal_windows = data.get("temporal_windows", {})
                self.last_semantic = datetime.fromisoformat(
                    data.get("last_semantic", datetime.now().isoformat())
                )
                self.last_salience = datetime.fromisoformat(
                    data.get("last_salience", datetime.now().isoformat())
                )
                self.last_optimization = datetime.fromisoformat(
                    data.get("last_optimization", datetime.now().isoformat())
                )
                self.last_adaptation = datetime.fromisoformat(
                    data.get("last_adaptation", datetime.now().isoformat())
                )

    async def get_novelty_stats(self) -> Dict[str, Any]:
        """Get statistics about novelty memory."""
        stats = {
            "total_items": len(self.items),
            "novelty_stats": {
                "average_novelty": sum(self.novelty_scores.values()) / len(self.novelty_scores) if self.novelty_scores else 0,
                "high_novelty_items": sum(1 for s in self.novelty_scores.values() if s > 0.7),
                "low_novelty_items": sum(1 for s in self.novelty_scores.values() if s < 0.3)
            },
            "salience_stats": {
                "average_salience": sum(self.salience_scores.values()) / len(self.salience_scores) if self.salience_scores else 0,
                "high_salience_items": sum(1 for s in self.salience_scores.values() if s > 0.7),
                "low_salience_items": sum(1 for s in self.salience_scores.values() if s < 0.3)
            },
            "pattern_stats": {
                "total_patterns": sum(
                    len(patterns) for patterns in self.pattern_matches.values()
                ),
                "average_patterns": sum(
                    len(patterns) for patterns in self.pattern_matches.values()
                ) / len(self.pattern_matches) if self.pattern_matches else 0
            },
            "semantic_stats": {
                "total_vectors": len(self.semantic_vectors),
                "vector_dimensions": len(next(iter(self.semantic_vectors.values()))) if self.semantic_vectors else 0
            },
            "concept_stats": {
                "total_concepts": sum(
                    len(concepts) for concepts in self.concept_maps.values()
                ),
                "average_concepts": sum(
                    len(concepts) for concepts in self.concept_maps.values()
                ) / len(self.concept_maps) if self.concept_maps else 0
            },
            "relation_stats": {
                "total_relations": sum(
                    len(relations) for relations in self.relation_graphs.values()
                ),
                "average_relations": sum(
                    len(relations) for relations in self.relation_graphs.values()
                ) / len(self.relation_graphs) if self.relation_graphs else 0
            },
            "temporal_stats": {
                "total_windows": len(self.temporal_windows),
                "average_window_size": sum(
                    len(window) for window in self.temporal_windows.values()
                ) / len(self.temporal_windows) if self.temporal_windows else 0
            }
        }
        
        return stats

    async def get_novelty_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for novelty optimization."""
        suggestions = []
        
        # Check item count
        if len(self.items) > self.max_items * 0.8:
            suggestions.append({
                "type": "item_limit",
                "suggestion": "Consider increasing max_items or removing less novel items"
            })
        
        # Check novelty distribution
        stats = await self.get_novelty_stats()
        if stats["novelty_stats"]["average_novelty"] < 0.5:
            suggestions.append({
                "type": "novelty_improvement",
                "suggestion": "Consider adjusting novelty thresholds or decay rates"
            })
        
        # Check salience distribution
        if stats["salience_stats"]["average_salience"] < 0.5:
            suggestions.append({
                "type": "salience_improvement",
                "suggestion": "Consider adjusting salience thresholds or decay rates"
            })
        
        # Check pattern coverage
        if stats["pattern_stats"]["average_patterns"] < 2:
            suggestions.append({
                "type": "pattern_enhancement",
                "suggestion": "Consider enhancing pattern detection"
            })
        
        # Check semantic coverage
        if stats["semantic_stats"]["total_vectors"] < len(self.items) * 0.8:
            suggestions.append({
                "type": "semantic_enhancement",
                "suggestion": "Consider improving semantic vector generation"
            })
        
        # Check concept coverage
        if stats["concept_stats"]["average_concepts"] < 2:
            suggestions.append({
                "type": "concept_enhancement",
                "suggestion": "Consider improving concept analysis"
            })
        
        # Check relation coverage
        if stats["relation_stats"]["average_relations"] < 2:
            suggestions.append({
                "type": "relation_enhancement",
                "suggestion": "Consider improving relation analysis"
            })
        
        # Check temporal coverage
        if stats["temporal_stats"]["average_window_size"] < 2:
            suggestions.append({
                "type": "temporal_enhancement",
                "suggestion": "Consider improving temporal analysis"
            })
        
        return suggestions 