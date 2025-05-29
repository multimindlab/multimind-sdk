"""
Active learning memory implementation.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class ActiveLearningMemory(BaseMemory):
    """Memory that implements active learning/reinforced memory."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_items: int = 1000,
        max_feedback: int = 10000,
        feedback_retention_days: int = 30,
        enable_feedback_analysis: bool = True,
        analysis_interval: int = 3600,  # 1 hour
        enable_reinforcement: bool = True,
        reinforcement_threshold: float = 0.7,
        enable_optimization: bool = True,
        optimization_interval: int = 3600  # 1 hour
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_items = max_items
        self.max_feedback = max_feedback
        self.feedback_retention_days = feedback_retention_days
        self.enable_feedback_analysis = enable_feedback_analysis
        self.analysis_interval = analysis_interval
        self.enable_reinforcement = enable_reinforcement
        self.reinforcement_threshold = reinforcement_threshold
        self.enable_optimization = enable_optimization
        self.optimization_interval = optimization_interval
        
        # Initialize storage
        self.items: List[Dict[str, Any]] = []
        self.feedback: List[Dict[str, Any]] = []  # Feedback log
        self.reinforcement: Dict[str, List[Dict[str, Any]]] = {}  # item_id -> reinforcement data
        self.last_analysis = datetime.now()
        self.last_optimization = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message and track feedback."""
        # Create new item
        item_id = f"item_{len(self.items)}"
        new_item = {
            "id": item_id,
            "content": message["content"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "type": "message",
                "created_at": datetime.now().isoformat(),
                "modified_at": datetime.now().isoformat(),
                "feedback_count": 0,
                "reinforcement_count": 0
            }
        }
        
        # Add to storage
        self.items.append(new_item)
        
        # Track feedback if needed
        if self.enable_reinforcement:
            await self._track_feedback(item_id, new_item)
        
        # Analyze feedback if needed
        if self.enable_feedback_analysis and (
            datetime.now() - self.last_analysis
        ).total_seconds() >= self.analysis_interval:
            await self._analyze_feedback()
        
        # Maintain item limit
        await self._maintain_item_limit()
        
        await self.save()

    async def _track_feedback(self, item_id: str, item: Dict[str, Any]) -> None:
        """Track feedback for a new item."""
        try:
            # Generate feedback analysis prompt
            prompt = f"""
            Analyze potential feedback for this item:
            
            {item['content']}
            
            Return a JSON object with:
            1. feedback_types: list of strings
            2. feedback_confidence: list of floats
            3. reinforcement_suggestions: list of strings
            """
            response = await self.llm.generate(prompt)
            feedback = json.loads(response)
            
            # Create feedback entries
            for i, feedback_type in enumerate(feedback["feedback_types"]):
                feedback_entry = {
                    "id": f"feedback_{len(self.feedback)}",
                    "item_id": item_id,
                    "type": feedback_type,
                    "confidence": feedback["feedback_confidence"][i],
                    "suggestion": feedback["reinforcement_suggestions"][i],
                    "timestamp": datetime.now().isoformat()
                }
                self.feedback.append(feedback_entry)
                
                # Update reinforcement data
                if item_id not in self.reinforcement:
                    self.reinforcement[item_id] = []
                self.reinforcement[item_id].append({
                    "feedback_id": feedback_entry["id"],
                    "type": feedback_type,
                    "confidence": feedback["feedback_confidence"][i],
                    "timestamp": feedback_entry["timestamp"]
                })
            
            # Update item metadata
            item["metadata"]["feedback_count"] = len(feedback["feedback_types"])
            item["metadata"]["reinforcement_count"] = len(feedback["reinforcement_suggestions"])
            
        except Exception as e:
            print(f"Error tracking feedback: {e}")

    async def _analyze_feedback(self) -> None:
        """Analyze feedback patterns and reinforcement."""
        # Group feedback by item
        item_feedback = {}
        for feedback in self.feedback:
            if feedback["item_id"] not in item_feedback:
                item_feedback[feedback["item_id"]] = []
            item_feedback[feedback["item_id"]].append(feedback)
        
        # Analyze each item's feedback
        for item_id, feedback_list in item_feedback.items():
            try:
                # Generate feedback analysis prompt
                prompt = f"""
                Analyze this feedback:
                
                {json.dumps(feedback_list, indent=2)}
                
                Return a JSON object with:
                1. feedback_patterns: list of strings
                2. reinforcement_quality: float (0-1)
                3. improvement_suggestions: list of strings
                """
                response = await self.llm.generate(prompt)
                analysis = json.loads(response)
                
                # Update reinforcement data
                if item_id in self.reinforcement:
                    self.reinforcement[item_id].append({
                        "type": "feedback_analysis",
                        "patterns": analysis["feedback_patterns"],
                        "quality": analysis["reinforcement_quality"],
                        "suggestions": analysis["improvement_suggestions"],
                        "timestamp": datetime.now().isoformat()
                    })
                
            except Exception as e:
                print(f"Error analyzing feedback: {e}")
        
        # Update last analysis time
        self.last_analysis = datetime.now()

    async def _maintain_item_limit(self) -> None:
        """Maintain item and feedback limits."""
        # Check item limit
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
        
        # Check feedback limit
        if len(self.feedback) > self.max_feedback:
            # Sort feedback by timestamp
            sorted_feedback = sorted(
                self.feedback,
                key=lambda x: datetime.fromisoformat(x["timestamp"])
            )
            
            # Remove oldest feedback
            self.feedback = sorted_feedback[len(self.feedback) - self.max_feedback:]

    async def _remove_item(self, item_id: str) -> None:
        """Remove an item and its associated feedback."""
        # Remove from items
        self.items = [i for i in self.items if i["id"] != item_id]
        
        # Remove associated feedback
        self.feedback = [f for f in self.feedback if f["item_id"] != item_id]
        
        # Remove from reinforcement
        if item_id in self.reinforcement:
            del self.reinforcement[item_id]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from all items."""
        messages = []
        for item in self.items:
            messages.append({
                "role": "active_learning_memory",
                "content": item["content"],
                "timestamp": item["timestamp"]
            })
        return sorted(messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear all items and feedback."""
        self.items = []
        self.feedback = []
        self.reinforcement = {}
        await self.save()

    async def save(self) -> None:
        """Save items and feedback to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "items": self.items,
                    "feedback": self.feedback,
                    "reinforcement": self.reinforcement,
                    "last_analysis": self.last_analysis.isoformat(),
                    "last_optimization": self.last_optimization.isoformat()
                }, f)

    def load(self) -> None:
        """Load items and feedback from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.items = data.get("items", [])
                self.feedback = data.get("feedback", [])
                self.reinforcement = data.get("reinforcement", {})
                self.last_analysis = datetime.fromisoformat(
                    data.get("last_analysis", datetime.now().isoformat())
                )
                self.last_optimization = datetime.fromisoformat(
                    data.get("last_optimization", datetime.now().isoformat())
                )

    async def get_active_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about active learning memory."""
        stats = {
            "total_items": len(self.items),
            "feedback_stats": {
                "total_feedback": len(self.feedback),
                "feedback_types": len(set(f["type"] for f in self.feedback)),
                "average_feedback_per_item": len(self.feedback) / len(self.items) if self.items else 0
            },
            "reinforcement_stats": {
                "total_reinforcement": sum(
                    len(reinforcement) for reinforcement in self.reinforcement.values()
                ),
                "average_reinforcement_per_item": sum(
                    len(reinforcement) for reinforcement in self.reinforcement.values()
                ) / len(self.reinforcement) if self.reinforcement else 0
            }
        }
        
        return stats

    async def get_active_learning_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for active learning memory optimization."""
        suggestions = []
        
        # Check item count
        if len(self.items) > self.max_items * 0.8:
            suggestions.append({
                "type": "item_limit",
                "suggestion": "Consider increasing max_items or removing older items"
            })
        
        # Check feedback count
        stats = await self.get_active_learning_stats()
        if stats["feedback_stats"]["total_feedback"] > self.max_feedback * 0.8:
            suggestions.append({
                "type": "feedback_limit",
                "suggestion": "Consider increasing max_feedback or compressing feedback"
            })
        
        # Check feedback coverage
        if stats["feedback_stats"]["average_feedback_per_item"] < 2:
            suggestions.append({
                "type": "feedback_coverage",
                "suggestion": "Consider improving feedback tracking"
            })
        
        # Check reinforcement quality
        if stats["reinforcement_stats"]["average_reinforcement_per_item"] < 2:
            suggestions.append({
                "type": "reinforcement_quality",
                "suggestion": "Consider improving reinforcement analysis"
            })
        
        return suggestions 