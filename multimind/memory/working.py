"""
Working memory implementation that manages temporary storage and manipulation of information.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class WorkingMemory(BaseMemory):
    """Memory that manages temporary storage and manipulation of information."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_items: int = 100,
        attention_threshold: float = 0.7,
        decay_rate: float = 0.95,
        decay_interval: int = 60,  # 1 minute
        attention_weights: Dict[str, float] = None,
        enable_consolidation: bool = True,
        consolidation_interval: int = 300,  # 5 minutes
        enable_attention: bool = True,
        attention_update_interval: int = 60,  # 1 minute
        min_attention: float = 0.1,
        enable_priority: bool = True,
        priority_weights: Dict[str, float] = None,
        enable_compression: bool = True,
        compression_threshold: float = 0.8,
        enable_backup: bool = True,
        backup_interval: int = 3600,  # 1 hour
        max_backups: int = 5
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_items = max_items
        self.attention_threshold = attention_threshold
        self.decay_rate = decay_rate
        self.decay_interval = decay_interval
        self.attention_weights = attention_weights or {
            "recency": 0.4,
            "relevance": 0.3,
            "importance": 0.3
        }
        self.enable_consolidation = enable_consolidation
        self.consolidation_interval = consolidation_interval
        self.enable_attention = enable_attention
        self.attention_update_interval = attention_update_interval
        self.min_attention = min_attention
        self.enable_priority = enable_priority
        self.priority_weights = priority_weights or {
            "urgency": 0.4,
            "importance": 0.3,
            "complexity": 0.3
        }
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.enable_backup = enable_backup
        self.backup_interval = backup_interval
        self.max_backups = max_backups
        
        # Initialize working memory storage
        self.items: List[Dict[str, Any]] = []
        self.item_embeddings: List[List[float]] = []
        self.attention_scores: Dict[str, float] = {}  # item_id -> attention score
        self.attention_history: Dict[str, List[Dict[str, Any]]] = {}  # item_id -> attention records
        self.consolidation_history: Dict[str, List[Dict[str, Any]]] = {}  # item_id -> consolidation records
        self.priority_scores: Dict[str, float] = {}  # item_id -> priority score
        self.compression_history: Dict[str, List[Dict[str, Any]]] = {}  # item_id -> compression records
        self.backup_history: List[Dict[str, Any]] = []  # List of backup records
        self.last_decay = datetime.now()
        self.last_consolidation = datetime.now()
        self.last_attention_update = datetime.now()
        self.last_backup = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message to working memory."""
        # Create new item
        item_id = f"item_{len(self.items)}"
        new_item = {
            "id": item_id,
            "content": message["content"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "type": "message",
                "importance": 1.0,
                "relevance": 1.0,
                "consolidated": False,
                "attention_score": 1.0,
                "decay_factor": 1.0,
                "priority_score": 1.0,
                "compressed": False,
                "compression_ratio": 1.0
            }
        }
        
        # Add to storage
        self.items.append(new_item)
        self.attention_scores[item_id] = 1.0
        self.attention_history[item_id] = []
        self.priority_scores[item_id] = 1.0
        self.compression_history[item_id] = []
        
        # Get item embedding
        embedding = await self.llm.embeddings(message["content"])
        self.item_embeddings.append(embedding)
        
        # Update attention scores
        if self.enable_attention:
            await self._update_attention_scores()
        
        # Update priority scores
        if self.enable_priority:
            await self._update_priority_scores()
        
        # Check for decay
        current_time = datetime.now()
        if (current_time - self.last_decay).total_seconds() > self.decay_interval:
            await self._apply_decay()
        
        # Check for consolidation
        if self.enable_consolidation:
            if (current_time - self.last_consolidation).total_seconds() > self.consolidation_interval:
                await self._consolidate_items()
        
        # Check for compression
        if self.enable_compression:
            await self._compress_items()
        
        # Check for backup
        if self.enable_backup:
            if (current_time - self.last_backup).total_seconds() > self.backup_interval:
                await self._create_backup()
        
        # Maintain item limit
        await self._maintain_item_limit()
        
        await self.save()

    async def _update_attention_scores(self) -> None:
        """Update attention scores for all items."""
        current_time = datetime.now()
        
        for item in self.items:
            item_id = item["id"]
            
            # Calculate attention components
            recency = self._calculate_recency(item["timestamp"])
            relevance = await self._calculate_relevance(item)
            importance = item["metadata"]["importance"]
            
            # Calculate weighted attention score
            attention_score = (
                self.attention_weights["recency"] * recency +
                self.attention_weights["relevance"] * relevance +
                self.attention_weights["importance"] * importance
            )
            
            # Update attention score
            self.attention_scores[item_id] = attention_score
            
            # Record attention update
            self.attention_history[item_id].append({
                "timestamp": current_time.isoformat(),
                "score": attention_score,
                "components": {
                    "recency": recency,
                    "relevance": relevance,
                    "importance": importance
                }
            })
        
        self.last_attention_update = current_time

    async def _update_priority_scores(self) -> None:
        """Update priority scores for all items."""
        for item in self.items:
            item_id = item["id"]
            
            # Calculate priority components
            urgency = self._calculate_urgency(item["timestamp"])
            importance = item["metadata"]["importance"]
            complexity = await self._calculate_complexity(item)
            
            # Calculate weighted priority score
            priority_score = (
                self.priority_weights["urgency"] * urgency +
                self.priority_weights["importance"] * importance +
                self.priority_weights["complexity"] * complexity
            )
            
            # Update priority score
            self.priority_scores[item_id] = priority_score
            item["metadata"]["priority_score"] = priority_score

    def _calculate_urgency(self, timestamp: str) -> float:
        """Calculate urgency score based on timestamp."""
        item_time = datetime.fromisoformat(timestamp)
        age = (datetime.now() - item_time).total_seconds()
        return np.exp(-age / 1800)  # Exponential decay over 30 minutes

    async def _calculate_complexity(self, item: Dict[str, Any]) -> float:
        """Calculate complexity score for an item."""
        try:
            # Count words and sentences
            words = len(item["content"].split())
            sentences = len(item["content"].split('.'))
            
            # Calculate complexity score
            complexity = (words / 100) * (sentences / 5)
            return min(1.0, complexity)
            
        except Exception as e:
            print(f"Error calculating complexity: {e}")
            return 0.5

    async def _compress_items(self) -> None:
        """Compress items to reduce memory usage."""
        for item in self.items:
            if item["metadata"]["compressed"]:
                continue
            
            try:
                # Generate compression prompt
                prompt = f"""
                Compress this information while maintaining key points:
                
                {item['content']}
                
                Return compressed version.
                """
                response = await self.llm.generate(prompt)
                
                # Calculate compression ratio
                original_length = len(item["content"])
                compressed_length = len(response)
                compression_ratio = compressed_length / original_length
                
                if compression_ratio <= self.compression_threshold:
                    # Update item
                    item["content"] = response
                    item["metadata"]["compressed"] = True
                    item["metadata"]["compression_ratio"] = compression_ratio
                    
                    # Record compression
                    self.compression_history[item["id"]].append({
                        "timestamp": datetime.now().isoformat(),
                        "original_length": original_length,
                        "compressed_length": compressed_length,
                        "compression_ratio": compression_ratio
                    })
                    
                    # Update embedding
                    idx = self.items.index(item)
                    self.item_embeddings[idx] = await self.llm.embeddings(response)
            
            except Exception as e:
                print(f"Error compressing item: {e}")

    async def _create_backup(self) -> None:
        """Create a backup of the current state."""
        backup = {
            "timestamp": datetime.now().isoformat(),
            "items": self.items,
            "attention_scores": self.attention_scores,
            "priority_scores": self.priority_scores
        }
        
        self.backup_history.append(backup)
        
        # Maintain backup limit
        if len(self.backup_history) > self.max_backups:
            self.backup_history.pop(0)
        
        self.last_backup = datetime.now()

    async def restore_from_backup(self, backup_index: int = -1) -> None:
        """Restore state from a backup."""
        if not self.backup_history:
            return
        
        backup = self.backup_history[backup_index]
        
        # Restore state
        self.items = backup["items"]
        self.attention_scores = backup["attention_scores"]
        self.priority_scores = backup["priority_scores"]
        
        # Recreate embeddings
        self.item_embeddings = []
        for item in self.items:
            self.item_embeddings.append(
                await self.llm.embeddings(item["content"])
            )
        
        await self.save()

    async def get_backup_info(self) -> List[Dict[str, Any]]:
        """Get information about available backups."""
        return [
            {
                "index": i,
                "timestamp": backup["timestamp"],
                "item_count": len(backup["items"])
            }
            for i, backup in enumerate(self.backup_history)
        ]

    async def get_working_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about working memory."""
        stats = await super().get_working_memory_stats()
        
        # Add priority statistics
        stats["priority_stats"] = {
            "average_priority": sum(self.priority_scores.values()) / len(self.priority_scores) if self.priority_scores else 0,
            "high_priority_items": sum(1 for score in self.priority_scores.values() if score > 0.7),
            "low_priority_items": sum(1 for score in self.priority_scores.values() if score < 0.3)
        }
        
        # Add compression statistics
        stats["compression_stats"] = {
            "compressed_items": sum(1 for item in self.items if item["metadata"]["compressed"]),
            "average_compression_ratio": sum(
                item["metadata"]["compression_ratio"]
                for item in self.items
                if item["metadata"]["compressed"]
            ) / sum(1 for item in self.items if item["metadata"]["compressed"])
            if any(item["metadata"]["compressed"] for item in self.items)
            else 0
        }
        
        # Add backup statistics
        stats["backup_stats"] = {
            "total_backups": len(self.backup_history),
            "latest_backup": self.backup_history[-1]["timestamp"] if self.backup_history else None,
            "backup_interval": self.backup_interval
        }
        
        return stats

    async def get_working_memory_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for working memory optimization."""
        suggestions = await super().get_working_memory_suggestions()
        
        # Add priority-related suggestions
        stats = await self.get_working_memory_stats()
        if stats["priority_stats"]["low_priority_items"] > len(self.items) * 0.3:
            suggestions.append({
                "type": "priority_management",
                "suggestion": "Consider removing or consolidating low-priority items"
            })
        
        # Add compression-related suggestions
        if stats["compression_stats"]["compressed_items"] < len(self.items) * 0.5:
            suggestions.append({
                "type": "compression",
                "suggestion": "Consider compressing more items to reduce memory usage"
            })
        
        # Add backup-related suggestions
        if not self.backup_history:
            suggestions.append({
                "type": "backup",
                "suggestion": "Consider creating regular backups of working memory"
            })
        
        return suggestions 