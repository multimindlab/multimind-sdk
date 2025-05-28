"""
Forgetting curve memory implementation based on Ebbinghaus's forgetting curve model.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class ForgettingCurveMemory(BaseMemory):
    """Memory that implements the Ebbinghaus forgetting curve model."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_items: int = 1000,
        initial_strength: float = 1.0,
        decay_rate: float = 0.1,  # Base decay rate
        retention_threshold: float = 0.3,  # Minimum strength to retain
        review_interval: int = 3600,  # 1 hour
        review_boost: float = 0.2,  # Strength boost on review
        enable_adaptive_decay: bool = True,
        adaptive_interval: int = 3600,  # 1 hour
        enable_spaced_repetition: bool = True,
        min_review_interval: int = 300,  # 5 minutes
        max_review_interval: int = 604800,  # 1 week
        enable_importance_weighting: bool = True,
        importance_threshold: float = 0.7,
        enable_consolidation: bool = True,
        consolidation_interval: int = 3600,  # 1 hour
        enable_learning_curve: bool = True,
        learning_rate: float = 0.1,
        enable_retrieval_practice: bool = True,
        retrieval_threshold: float = 0.5,
        enable_interference_analysis: bool = True,
        interference_threshold: float = 0.6,
        enable_optimization: bool = True,
        optimization_interval: int = 3600  # 1 hour
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_items = max_items
        self.initial_strength = initial_strength
        self.decay_rate = decay_rate
        self.retention_threshold = retention_threshold
        self.review_interval = review_interval
        self.review_boost = review_boost
        self.enable_adaptive_decay = enable_adaptive_decay
        self.adaptive_interval = adaptive_interval
        self.enable_spaced_repetition = enable_spaced_repetition
        self.min_review_interval = min_review_interval
        self.max_review_interval = max_review_interval
        self.enable_importance_weighting = enable_importance_weighting
        self.importance_threshold = importance_threshold
        self.enable_consolidation = enable_consolidation
        self.consolidation_interval = consolidation_interval
        self.enable_learning_curve = enable_learning_curve
        self.learning_rate = learning_rate
        self.enable_retrieval_practice = enable_retrieval_practice
        self.retrieval_threshold = retrieval_threshold
        self.enable_interference_analysis = enable_interference_analysis
        self.interference_threshold = interference_threshold
        self.enable_optimization = enable_optimization
        self.optimization_interval = optimization_interval
        
        # Initialize storage
        self.items: List[Dict[str, Any]] = []
        self.strengths: Dict[str, float] = {}  # item_id -> strength
        self.review_history: Dict[str, List[Dict[str, Any]]] = {}  # item_id -> review records
        self.learning_curves: Dict[str, List[Dict[str, Any]]] = {}  # item_id -> learning curve data
        self.interference_graph: Dict[str, Set[str]] = {}  # item_id -> interfering items
        self.last_review = datetime.now()
        self.last_adaptive = datetime.now()
        self.last_consolidation = datetime.now()
        self.last_optimization = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message and initialize forgetting curve."""
        # Create new item
        item_id = f"item_{len(self.items)}"
        new_item = {
            "id": item_id,
            "content": message["content"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "type": "message",
                "strength": self.initial_strength,
                "importance": 0.0,
                "review_count": 0,
                "last_review": None,
                "next_review": None,
                "learning_progress": 0.0,
                "interference_score": 0.0,
                "consolidation_score": 0.0,
                "optimization_score": 0.0
            }
        }
        
        # Add to storage
        self.items.append(new_item)
        self.strengths[item_id] = self.initial_strength
        
        # Initialize review history
        self.review_history[item_id] = []
        
        # Initialize learning curve
        self.learning_curves[item_id] = []
        
        # Initialize interference graph
        self.interference_graph[item_id] = set()
        
        # Calculate initial importance
        if self.enable_importance_weighting:
            await self._calculate_importance(item_id)
        
        # Schedule first review
        if self.enable_spaced_repetition:
            await self._schedule_review(item_id)
        
        # Analyze interference
        if self.enable_interference_analysis:
            await self._analyze_interference(item_id)
        
        # Update learning curve
        if self.enable_learning_curve:
            await self._update_learning_curve(item_id)
        
        # Maintain item limit
        await self._maintain_item_limit()
        
        await self.save()

    async def _calculate_importance(self, item_id: str) -> None:
        """Calculate importance score for an item."""
        item = next(i for i in self.items if i["id"] == item_id)
        
        try:
            # Generate importance analysis prompt
            prompt = f"""
            Analyze the importance of this item:
            
            {item['content']}
            
            Return a JSON object with:
            1. importance_score: float (0-1)
            2. importance_factors: list of strings
            3. importance_reason: string
            """
            response = await self.llm.generate(prompt)
            importance = json.loads(response)
            
            # Update item metadata
            item["metadata"]["importance"] = importance["importance_score"]
            
        except Exception as e:
            print(f"Error calculating importance: {e}")

    async def _schedule_review(self, item_id: str) -> None:
        """Schedule next review using spaced repetition."""
        item = next(i for i in self.items if i["id"] == item_id)
        review_count = item["metadata"]["review_count"]
        
        # Calculate next review interval using exponential spacing
        base_interval = self.min_review_interval
        max_interval = self.max_review_interval
        
        # Adjust interval based on review count and importance
        importance_factor = item["metadata"]["importance"]
        interval = min(
            max_interval,
            base_interval * (2 ** review_count) * (1 + importance_factor)
        )
        
        # Schedule next review
        next_review = datetime.now() + timedelta(seconds=interval)
        item["metadata"]["next_review"] = next_review.isoformat()

    async def _analyze_interference(self, item_id: str) -> None:
        """Analyze potential interference with other items."""
        item = next(i for i in self.items if i["id"] == item_id)
        
        try:
            # Generate interference analysis prompt
            prompt = f"""
            Analyze potential interference with this item:
            
            {item['content']}
            
            Return a JSON object with:
            1. interference_score: float (0-1)
            2. interfering_items: list of strings
            3. interference_type: string
            4. interference_reason: string
            """
            response = await self.llm.generate(prompt)
            interference = json.loads(response)
            
            # Update interference graph
            for interfering_item in interference["interfering_items"]:
                self.interference_graph[item_id].add(interfering_item)
            
            # Update item metadata
            item["metadata"]["interference_score"] = interference["interference_score"]
            
        except Exception as e:
            print(f"Error analyzing interference: {e}")

    async def _update_learning_curve(self, item_id: str) -> None:
        """Update learning curve for an item."""
        item = next(i for i in self.items if i["id"] == item_id)
        
        # Calculate learning progress
        strength = self.strengths[item_id]
        importance = item["metadata"]["importance"]
        review_count = item["metadata"]["review_count"]
        
        # Update learning progress
        progress = min(
            1.0,
            item["metadata"]["learning_progress"] +
            self.learning_rate * (strength * importance * (1 + 0.1 * review_count))
        )
        
        item["metadata"]["learning_progress"] = progress
        
        # Record learning curve point
        self.learning_curves[item_id].append({
            "timestamp": datetime.now().isoformat(),
            "strength": strength,
            "importance": importance,
            "review_count": review_count,
            "progress": progress
        })

    async def _update_strength(self, item_id: str) -> None:
        """Update memory strength based on forgetting curve."""
        item = next(i for i in self.items if i["id"] == item_id)
        current_strength = self.strengths[item_id]
        
        # Calculate time since last review
        last_review = datetime.fromisoformat(
            item["metadata"]["last_review"] or item["timestamp"]
        )
        time_diff = (datetime.now() - last_review).total_seconds()
        
        # Calculate decay factor
        decay_factor = np.exp(-self.decay_rate * time_diff)
        
        # Apply importance weighting
        importance_factor = item["metadata"]["importance"]
        
        # Calculate new strength
        new_strength = current_strength * decay_factor * (1 + 0.2 * importance_factor)
        
        # Update strength
        self.strengths[item_id] = max(0.0, min(1.0, new_strength))
        
        # Update item metadata
        item["metadata"]["strength"] = new_strength

    async def _review_item(self, item_id: str) -> None:
        """Review an item and update its strength."""
        item = next(i for i in self.items if i["id"] == item_id)
        
        # Update strength
        await self._update_strength(item_id)
        
        # Apply review boost
        current_strength = self.strengths[item_id]
        new_strength = min(1.0, current_strength + self.review_boost)
        self.strengths[item_id] = new_strength
        
        # Update review count
        item["metadata"]["review_count"] += 1
        
        # Update last review timestamp
        item["metadata"]["last_review"] = datetime.now().isoformat()
        
        # Record review
        self.review_history[item_id].append({
            "timestamp": datetime.now().isoformat(),
            "strength_before": current_strength,
            "strength_after": new_strength,
            "review_count": item["metadata"]["review_count"]
        })
        
        # Schedule next review
        if self.enable_spaced_repetition:
            await self._schedule_review(item_id)
        
        # Update learning curve
        if self.enable_learning_curve:
            await self._update_learning_curve(item_id)

    async def _maintain_item_limit(self) -> None:
        """Maintain item limit by removing weakest items."""
        if len(self.items) > self.max_items:
            # Sort items by strength and importance
            sorted_items = sorted(
                self.items,
                key=lambda x: (
                    self.strengths[x["id"]] *
                    x["metadata"]["importance"]
                )
            )
            
            # Remove weakest items
            items_to_remove = sorted_items[:len(self.items) - self.max_items]
            for item in items_to_remove:
                await self._remove_item(item["id"])

    async def _remove_item(self, item_id: str) -> None:
        """Remove an item and its associated data."""
        # Remove from items
        self.items = [i for i in self.items if i["id"] != item_id]
        
        # Remove from strengths
        if item_id in self.strengths:
            del self.strengths[item_id]
        
        # Remove from review history
        if item_id in self.review_history:
            del self.review_history[item_id]
        
        # Remove from learning curves
        if item_id in self.learning_curves:
            del self.learning_curves[item_id]
        
        # Remove from interference graph
        if item_id in self.interference_graph:
            del self.interference_graph[item_id]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from all items."""
        messages = []
        for item in self.items:
            messages.append({
                "role": "forgetting_curve_memory",
                "content": item["content"],
                "timestamp": item["timestamp"]
            })
        return sorted(messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear all items."""
        self.items = []
        self.strengths = {}
        self.review_history = {}
        self.learning_curves = {}
        self.interference_graph = {}
        await self.save()

    async def save(self) -> None:
        """Save items to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "items": self.items,
                    "strengths": self.strengths,
                    "review_history": self.review_history,
                    "learning_curves": self.learning_curves,
                    "interference_graph": {
                        k: list(v) for k, v in self.interference_graph.items()
                    },
                    "last_review": self.last_review.isoformat(),
                    "last_adaptive": self.last_adaptive.isoformat(),
                    "last_consolidation": self.last_consolidation.isoformat(),
                    "last_optimization": self.last_optimization.isoformat()
                }, f)

    def load(self) -> None:
        """Load items from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.items = data.get("items", [])
                self.strengths = data.get("strengths", {})
                self.review_history = data.get("review_history", {})
                self.learning_curves = data.get("learning_curves", {})
                self.interference_graph = {
                    k: set(v) for k, v in data.get("interference_graph", {}).items()
                }
                self.last_review = datetime.fromisoformat(
                    data.get("last_review", datetime.now().isoformat())
                )
                self.last_adaptive = datetime.fromisoformat(
                    data.get("last_adaptive", datetime.now().isoformat())
                )
                self.last_consolidation = datetime.fromisoformat(
                    data.get("last_consolidation", datetime.now().isoformat())
                )
                self.last_optimization = datetime.fromisoformat(
                    data.get("last_optimization", datetime.now().isoformat())
                )

    async def get_forgetting_curve_stats(self) -> Dict[str, Any]:
        """Get statistics about forgetting curve memory."""
        stats = {
            "total_items": len(self.items),
            "strength_stats": {
                "average_strength": sum(self.strengths.values()) / len(self.strengths) if self.strengths else 0,
                "strong_items": sum(1 for s in self.strengths.values() if s > 0.7),
                "weak_items": sum(1 for s in self.strengths.values() if s < 0.3)
            },
            "review_stats": {
                "total_reviews": sum(
                    len(reviews) for reviews in self.review_history.values()
                ),
                "average_reviews": sum(
                    len(reviews) for reviews in self.review_history.values()
                ) / len(self.review_history) if self.review_history else 0
            },
            "learning_stats": {
                "average_progress": sum(
                    item["metadata"]["learning_progress"]
                    for item in self.items
                ) / len(self.items) if self.items else 0,
                "items_with_progress": sum(
                    1 for item in self.items
                    if item["metadata"]["learning_progress"] > 0
                )
            },
            "interference_stats": {
                "total_interferences": sum(
                    len(interferences)
                    for interferences in self.interference_graph.values()
                ),
                "average_interference": sum(
                    len(interferences)
                    for interferences in self.interference_graph.values()
                ) / len(self.interference_graph) if self.interference_graph else 0
            }
        }
        
        return stats

    async def get_forgetting_curve_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for forgetting curve optimization."""
        suggestions = []
        
        # Check item count
        if len(self.items) > self.max_items * 0.8:
            suggestions.append({
                "type": "item_limit",
                "suggestion": "Consider increasing max_items or removing weaker items"
            })
        
        # Check strength distribution
        stats = await self.get_forgetting_curve_stats()
        if stats["strength_stats"]["average_strength"] < 0.5:
            suggestions.append({
                "type": "strength_improvement",
                "suggestion": "Consider increasing review frequency or decay rate"
            })
        
        # Check review coverage
        if stats["review_stats"]["average_reviews"] < 2:
            suggestions.append({
                "type": "review_coverage",
                "suggestion": "Consider increasing review frequency"
            })
        
        # Check learning progress
        if stats["learning_stats"]["average_progress"] < 0.5:
            suggestions.append({
                "type": "learning_enhancement",
                "suggestion": "Consider enhancing learning mechanisms"
            })
        
        # Check interference
        if stats["interference_stats"]["average_interference"] > 3:
            suggestions.append({
                "type": "interference_reduction",
                "suggestion": "Consider reducing interference between items"
            })
        
        return suggestions 