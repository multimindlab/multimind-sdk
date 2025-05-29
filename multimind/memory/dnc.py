"""
Differentiable Neural Computer (DNC) memory implementation.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class DNCMemory(BaseMemory):
    """Memory that implements Differentiable Neural Computer architecture."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        memory_size: int = 1000,
        word_size: int = 64,
        num_read_heads: int = 4,
        num_write_heads: int = 1,
        controller_size: int = 256,
        enable_content_lookup: bool = True,
        enable_temporal_linkage: bool = True,
        enable_dynamic_allocation: bool = True,
        enable_usage_tracking: bool = True,
        enable_attention: bool = True,
        enable_learning: bool = True,
        learning_rate: float = 0.01,
        enable_optimization: bool = True,
        optimization_interval: int = 3600,  # 1 hour
        enable_analysis: bool = True,
        analysis_interval: int = 3600,  # 1 hour
        enable_compression: bool = True,
        compression_threshold: float = 0.8,
        enable_backup: bool = True,
        backup_interval: int = 3600,  # 1 hour
        max_backups: int = 24
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.controller_size = controller_size
        self.enable_content_lookup = enable_content_lookup
        self.enable_temporal_linkage = enable_temporal_linkage
        self.enable_dynamic_allocation = enable_dynamic_allocation
        self.enable_usage_tracking = enable_usage_tracking
        self.enable_attention = enable_attention
        self.enable_learning = enable_learning
        self.learning_rate = learning_rate
        self.enable_optimization = enable_optimization
        self.optimization_interval = optimization_interval
        self.enable_analysis = enable_analysis
        self.analysis_interval = analysis_interval
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.enable_backup = enable_backup
        self.backup_interval = backup_interval
        self.max_backups = max_backups

        # Initialize DNC memory matrix
        self.memory_matrix = np.zeros((memory_size, word_size))
        self.usage_vector = np.zeros(memory_size)
        self.precedence_vector = np.zeros(memory_size)
        self.link_matrix = np.zeros((memory_size, memory_size))
        self.write_weighting = np.zeros((num_write_heads, memory_size))
        self.read_weighting = np.zeros((num_read_heads, memory_size))
        self.read_vectors = np.zeros((num_read_heads, word_size))
        self.controller_state = np.zeros(controller_size)

        # Initialize storage
        self.items: List[Dict[str, Any]] = []
        self.item_embeddings: List[List[float]] = []
        self.attention_scores: Dict[str, float] = {}  # item_id -> attention score
        self.usage_history: Dict[str, List[Dict[str, Any]]] = {}  # item_id -> usage records
        self.learning_history: Dict[str, List[Dict[str, Any]]] = {}  # item_id -> learning records
        self.optimization_history: List[Dict[str, Any]] = []  # List of optimization records
        self.analysis_history: List[Dict[str, Any]] = []  # List of analysis records
        self.backup_history: List[Dict[str, Any]] = []  # List of backup records
        self.last_optimization = datetime.now()
        self.last_analysis = datetime.now()
        self.last_backup = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message to DNC memory."""
        # Create new item
        item_id = f"item_{len(self.items)}"
        new_item = {
            "id": item_id,
            "content": message["content"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "type": "message",
                "usage_count": 0,
                "last_accessed": datetime.now().isoformat(),
                "attention_score": 0.0,
                "learning_progress": 0.0,
                "compression_ratio": 1.0,
                "memory_location": None,
                "read_heads": [],
                "write_heads": []
            }
        }

        # Get item embedding
        embedding = await self.llm.embeddings(message["content"])
        self.item_embeddings.append(embedding)

        # Update memory matrix
        await self._update_memory_matrix(new_item, embedding)

        # Update attention if enabled
        if self.enable_attention:
            await self._update_attention(new_item)

        # Update learning if enabled
        if self.enable_learning:
            await self._update_learning(new_item)

        # Check for optimization
        if self.enable_optimization:
            current_time = datetime.now()
            if (current_time - self.last_optimization).total_seconds() > self.optimization_interval:
                await self._optimize_memory()

        # Check for analysis
        if self.enable_analysis:
            if (current_time - self.last_analysis).total_seconds() > self.analysis_interval:
                await self._analyze_memory()

        # Check for backup
        if self.enable_backup:
            if (current_time - self.last_backup).total_seconds() > self.backup_interval:
                await self._create_backup()

        # Add to storage
        self.items.append(new_item)
        await self.save()

    async def _update_memory_matrix(self, item: Dict[str, Any], embedding: List[float]) -> None:
        """Update DNC memory matrix with new item."""
        try:
            # Find available memory location
            if self.enable_dynamic_allocation:
                location = self._find_available_location()
            else:
                location = len(self.items) % self.memory_size

            # Update memory matrix
            self.memory_matrix[location] = np.array(embedding)
            item["metadata"]["memory_location"] = location

            # Update usage vector
            if self.enable_usage_tracking:
                self.usage_vector[location] += 1
                self.usage_history[item["id"]] = [{
                    "timestamp": datetime.now().isoformat(),
                    "location": location,
                    "usage_count": self.usage_vector[location]
                }]

            # Update link matrix if temporal linkage is enabled
            if self.enable_temporal_linkage and len(self.items) > 0:
                prev_location = self.items[-1]["metadata"]["memory_location"]
                self.link_matrix[prev_location, location] = 1
                self.precedence_vector[location] = 1

            # Update write weighting
            self.write_weighting[0, location] = 1

            # Update read weighting for content lookup
            if self.enable_content_lookup:
                await self._update_read_weighting(item, embedding)

        except Exception as e:
            print(f"Error updating memory matrix: {e}")

    def _find_available_location(self) -> int:
        """Find available memory location using dynamic allocation."""
        # Find least used memory location
        if self.enable_usage_tracking:
            return np.argmin(self.usage_vector)
        else:
            # Use first available location
            return len(self.items) % self.memory_size

    async def _update_read_weighting(self, item: Dict[str, Any], embedding: List[float]) -> None:
        """Update read weighting based on content similarity."""
        try:
            # Calculate similarity with existing items
            similarities = []
            for i, existing_embedding in enumerate(self.item_embeddings):
                similarity = np.dot(embedding, existing_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(existing_embedding)
                )
                similarities.append((i, similarity))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Update read weighting for top similar items
            for head_idx in range(self.num_read_heads):
                if head_idx < len(similarities):
                    item_idx, similarity = similarities[head_idx]
                    location = self.items[item_idx]["metadata"]["memory_location"]
                    self.read_weighting[head_idx, location] = similarity
                    item["metadata"]["read_heads"].append({
                        "head": head_idx,
                        "location": location,
                        "similarity": similarity
                    })

        except Exception as e:
            print(f"Error updating read weighting: {e}")

    async def _update_attention(self, item: Dict[str, Any]) -> None:
        """Update attention scores for items."""
        try:
            # Calculate attention score based on usage and recency
            usage_score = item["metadata"]["usage_count"] / max(1, max(self.usage_vector))
            recency_score = 1.0  # New items get full recency score
            attention_score = 0.7 * usage_score + 0.3 * recency_score

            # Update attention score
            self.attention_scores[item["id"]] = attention_score
            item["metadata"]["attention_score"] = attention_score

        except Exception as e:
            print(f"Error updating attention: {e}")

    async def _update_learning(self, item: Dict[str, Any]) -> None:
        """Update learning progress for items."""
        try:
            # Calculate learning progress based on usage and attention
            usage_count = item["metadata"]["usage_count"]
            attention_score = item["metadata"]["attention_score"]
            learning_progress = min(1.0, (usage_count * self.learning_rate + attention_score) / 2)

            # Update learning progress
            item["metadata"]["learning_progress"] = learning_progress
            self.learning_history[item["id"]] = [{
                "timestamp": datetime.now().isoformat(),
                "progress": learning_progress,
                "usage_count": usage_count,
                "attention_score": attention_score
            }]

        except Exception as e:
            print(f"Error updating learning: {e}")

    async def _optimize_memory(self) -> None:
        """Optimize memory matrix and weightings."""
        try:
            # Compress memory if enabled
            if self.enable_compression:
                await self._compress_memory()

            # Update controller state
            self.controller_state = np.mean(self.memory_matrix, axis=0)

            # Record optimization
            self.optimization_history.append({
                "timestamp": datetime.now().isoformat(),
                "memory_usage": np.mean(self.usage_vector),
                "controller_state": self.controller_state.tolist()
            })

            self.last_optimization = datetime.now()

        except Exception as e:
            print(f"Error optimizing memory: {e}")

    async def _compress_memory(self) -> None:
        """Compress memory matrix to reduce size."""
        try:
            # Calculate compression ratio
            original_size = self.memory_matrix.nbytes
            compressed_matrix = np.zeros_like(self.memory_matrix)

            # Compress each row
            for i in range(self.memory_size):
                if np.any(self.memory_matrix[i]):
                    # Use PCA or similar for compression
                    compressed_matrix[i] = self.memory_matrix[i] * self.compression_threshold

            # Update memory matrix if compression is significant
            compressed_size = compressed_matrix.nbytes
            ratio = compressed_size / original_size

            if ratio < self.compression_threshold:
                self.memory_matrix = compressed_matrix

        except Exception as e:
            print(f"Error compressing memory: {e}")

    async def _analyze_memory(self) -> None:
        """Analyze memory state and patterns."""
        try:
            # Generate analysis prompt
            prompt = f"""
            Analyze DNC memory state:
            
            Memory size: {self.memory_size}
            Word size: {self.word_size}
            Read heads: {self.num_read_heads}
            Write heads: {self.num_write_heads}
            
            Memory usage: {np.mean(self.usage_vector):.2f}
            Controller state: {self.controller_state.tolist()}
            
            Return a JSON object with:
            1. analysis: dict of string -> any
            2. suggestions: list of string
            3. metrics: dict of string -> float
            """
            response = await self.llm.generate(prompt)
            analysis = json.loads(response)

            # Record analysis
            self.analysis_history.append({
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis["analysis"],
                "suggestions": analysis["suggestions"],
                "metrics": analysis["metrics"]
            })

            self.last_analysis = datetime.now()

        except Exception as e:
            print(f"Error analyzing memory: {e}")

    async def _create_backup(self) -> None:
        """Create backup of memory state."""
        try:
            backup = {
                "timestamp": datetime.now().isoformat(),
                "memory_matrix": self.memory_matrix.tolist(),
                "usage_vector": self.usage_vector.tolist(),
                "precedence_vector": self.precedence_vector.tolist(),
                "link_matrix": self.link_matrix.tolist(),
                "write_weighting": self.write_weighting.tolist(),
                "read_weighting": self.read_weighting.tolist(),
                "read_vectors": self.read_vectors.tolist(),
                "controller_state": self.controller_state.tolist(),
                "items": self.items,
                "item_embeddings": self.item_embeddings,
                "attention_scores": self.attention_scores,
                "usage_history": self.usage_history,
                "learning_history": self.learning_history
            }

            self.backup_history.append(backup)

            # Maintain backup limit
            if len(self.backup_history) > self.max_backups:
                self.backup_history = self.backup_history[-self.max_backups:]

            self.last_backup = datetime.now()

        except Exception as e:
            print(f"Error creating backup: {e}")

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from memory."""
        messages = []
        for item in self.items:
            messages.append({
                "role": "dnc_memory",
                "content": item["content"],
                "timestamp": item["timestamp"]
            })
        return sorted(messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear all memory."""
        self.memory_matrix = np.zeros((self.memory_size, self.word_size))
        self.usage_vector = np.zeros(self.memory_size)
        self.precedence_vector = np.zeros(self.memory_size)
        self.link_matrix = np.zeros((self.memory_size, self.memory_size))
        self.write_weighting = np.zeros((self.num_write_heads, self.memory_size))
        self.read_weighting = np.zeros((self.num_read_heads, self.memory_size))
        self.read_vectors = np.zeros((self.num_read_heads, self.word_size))
        self.controller_state = np.zeros(self.controller_size)
        self.items = []
        self.item_embeddings = []
        self.attention_scores = {}
        self.usage_history = {}
        self.learning_history = {}
        self.optimization_history = []
        self.analysis_history = []
        self.backup_history = []
        await self.save()

    async def save(self) -> None:
        """Save memory to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "memory_matrix": self.memory_matrix.tolist(),
                    "usage_vector": self.usage_vector.tolist(),
                    "precedence_vector": self.precedence_vector.tolist(),
                    "link_matrix": self.link_matrix.tolist(),
                    "write_weighting": self.write_weighting.tolist(),
                    "read_weighting": self.read_weighting.tolist(),
                    "read_vectors": self.read_vectors.tolist(),
                    "controller_state": self.controller_state.tolist(),
                    "items": self.items,
                    "item_embeddings": self.item_embeddings,
                    "attention_scores": self.attention_scores,
                    "usage_history": self.usage_history,
                    "learning_history": self.learning_history,
                    "optimization_history": self.optimization_history,
                    "analysis_history": self.analysis_history,
                    "backup_history": self.backup_history,
                    "last_optimization": self.last_optimization.isoformat(),
                    "last_analysis": self.last_analysis.isoformat(),
                    "last_backup": self.last_backup.isoformat()
                }, f)

    def load(self) -> None:
        """Load memory from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.memory_matrix = np.array(data.get("memory_matrix", []))
                self.usage_vector = np.array(data.get("usage_vector", []))
                self.precedence_vector = np.array(data.get("precedence_vector", []))
                self.link_matrix = np.array(data.get("link_matrix", []))
                self.write_weighting = np.array(data.get("write_weighting", []))
                self.read_weighting = np.array(data.get("read_weighting", []))
                self.read_vectors = np.array(data.get("read_vectors", []))
                self.controller_state = np.array(data.get("controller_state", []))
                self.items = data.get("items", [])
                self.item_embeddings = data.get("item_embeddings", [])
                self.attention_scores = data.get("attention_scores", {})
                self.usage_history = data.get("usage_history", {})
                self.learning_history = data.get("learning_history", {})
                self.optimization_history = data.get("optimization_history", [])
                self.analysis_history = data.get("analysis_history", [])
                self.backup_history = data.get("backup_history", [])
                self.last_optimization = datetime.fromisoformat(
                    data.get("last_optimization", datetime.now().isoformat())
                )
                self.last_analysis = datetime.fromisoformat(
                    data.get("last_analysis", datetime.now().isoformat())
                )
                self.last_backup = datetime.fromisoformat(
                    data.get("last_backup", datetime.now().isoformat())
                )

    async def get_dnc_stats(self) -> Dict[str, Any]:
        """Get statistics about DNC memory."""
        stats = {
            "memory_stats": {
                "total_items": len(self.items),
                "memory_usage": float(np.mean(self.usage_vector)),
                "memory_density": float(np.count_nonzero(self.memory_matrix) / self.memory_matrix.size),
                "controller_state": self.controller_state.tolist()
            },
            "attention_stats": {
                "total_attention_scores": len(self.attention_scores),
                "average_attention": sum(self.attention_scores.values()) / len(self.attention_scores) if self.attention_scores else 0,
                "max_attention": max(self.attention_scores.values()) if self.attention_scores else 0
            },
            "learning_stats": {
                "total_learning_records": sum(len(records) for records in self.learning_history.values()),
                "average_progress": sum(
                    record["progress"] for records in self.learning_history.values()
                    for record in records
                ) / sum(len(records) for records in self.learning_history.values()) if self.learning_history else 0
            },
            "optimization_stats": {
                "total_optimizations": len(self.optimization_history),
                "latest_optimization": self.optimization_history[-1]["timestamp"] if self.optimization_history else None,
                "optimization_frequency": self.optimization_interval
            },
            "analysis_stats": {
                "total_analyses": len(self.analysis_history),
                "latest_analysis": self.analysis_history[-1]["timestamp"] if self.analysis_history else None,
                "analysis_frequency": self.analysis_interval
            },
            "backup_stats": {
                "total_backups": len(self.backup_history),
                "latest_backup": self.backup_history[-1]["timestamp"] if self.backup_history else None,
                "backup_frequency": self.backup_interval
            }
        }
        return stats

    async def get_dnc_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for DNC memory optimization."""
        suggestions = []
        
        # Check memory usage
        if np.mean(self.usage_vector) > 0.8:
            suggestions.append({
                "type": "memory_usage",
                "suggestion": "Consider increasing memory size or implementing more aggressive compression"
            })
        
        # Check attention distribution
        if self.attention_scores:
            attention_values = list(self.attention_scores.values())
            if np.std(attention_values) < 0.1:
                suggestions.append({
                    "type": "attention_distribution",
                    "suggestion": "Consider adjusting attention scoring to better differentiate items"
                })
        
        # Check learning progress
        if self.learning_history:
            avg_progress = sum(
                record["progress"] for records in self.learning_history.values()
                for record in records
            ) / sum(len(records) for records in self.learning_history.values())
            if avg_progress < 0.3:
                suggestions.append({
                    "type": "learning_rate",
                    "suggestion": "Consider increasing learning rate or improving learning mechanisms"
                })
        
        # Check optimization frequency
        if len(self.optimization_history) < 2:
            suggestions.append({
                "type": "optimization_frequency",
                "suggestion": "Consider adjusting optimization interval"
            })
        
        # Check analysis coverage
        if len(self.analysis_history) < 2:
            suggestions.append({
                "type": "analysis_frequency",
                "suggestion": "Consider adjusting analysis interval"
            })
        
        # Check backup coverage
        if len(self.backup_history) < 2:
            suggestions.append({
                "type": "backup_frequency",
                "suggestion": "Consider adjusting backup interval"
            })
        
        return suggestions 