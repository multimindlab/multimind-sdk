"""
Time-weighted memory implementation that weights messages based on recency.
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import json
from pathlib import Path
import math
from .base import BaseMemory

class TimeWeightedMemory(BaseMemory):
    """Memory that weights messages based on their recency."""

    def __init__(
        self,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        decay_rate: float = 0.1,  # Exponential decay rate
        max_age_days: int = 30,  # Maximum age of messages to keep
        min_weight: float = 0.1,  # Minimum weight for messages
        decay_function: str = "exponential",  # Type of decay function
        time_units: str = "days"  # Time units for decay
    ):
        super().__init__(memory_key)
        self.storage_path = Path(storage_path) if storage_path else None
        self.decay_rate = decay_rate
        self.max_age_days = max_age_days
        self.min_weight = min_weight
        self.decay_function = decay_function
        self.time_units = time_units
        self.messages: List[Dict[str, Any]] = []
        self.load()

    def add_message(self, message: Dict[str, str]) -> None:
        """Add message with timestamp and weight."""
        message_with_metadata = {
            **message,
            "timestamp": datetime.now().isoformat(),
            "weight": 1.0,  # Initial weight for new messages
            "importance": 1.0  # Initial importance score
        }
        self.messages.append(message_with_metadata)
        self._update_weights()
        self.save()

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages with their current weights."""
        self._update_weights()  # Update weights before returning
        return self.messages

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        self.save()

    def save(self) -> None:
        """Save messages to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self.messages, f)

    def load(self) -> None:
        """Load messages from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                self.messages = json.load(f)

    def _get_decay_function(self) -> Callable[[float], float]:
        """Get the appropriate decay function."""
        if self.decay_function == "exponential":
            return lambda x: math.exp(-self.decay_rate * x)
        elif self.decay_function == "linear":
            return lambda x: max(0, 1 - (self.decay_rate * x))
        elif self.decay_function == "polynomial":
            return lambda x: 1 / (1 + self.decay_rate * x)
        else:
            return lambda x: math.exp(-self.decay_rate * x)

    def _convert_time_units(self, time_delta: timedelta) -> float:
        """Convert time delta to the specified units."""
        if self.time_units == "days":
            return time_delta.days + time_delta.seconds / (24 * 3600)
        elif self.time_units == "hours":
            return time_delta.days * 24 + time_delta.seconds / 3600
        elif self.time_units == "minutes":
            return time_delta.days * 24 * 60 + time_delta.seconds / 60
        else:
            return time_delta.days

    def _update_weights(self) -> None:
        """Update weights of all messages based on their age."""
        current_time = datetime.now()
        decay_func = self._get_decay_function()
        
        # Update weights and remove old messages
        self.messages = [
            msg for msg in self.messages
            if self._is_message_valid(msg, current_time, decay_func)
        ]
        
        # Sort by timestamp
        self.messages.sort(key=lambda x: x["timestamp"])

    def _is_message_valid(
        self,
        message: Dict[str, Any],
        current_time: datetime,
        decay_func: Callable[[float], float]
    ) -> bool:
        """Check if message is still valid based on age and weight."""
        msg_time = datetime.fromisoformat(message["timestamp"])
        time_delta = current_time - msg_time
        
        # Convert to appropriate time units
        age = self._convert_time_units(time_delta)
        
        # Remove messages older than max_age_days
        if age > self.max_age_days:
            return False
        
        # Calculate weight based on age and importance
        base_weight = decay_func(age)
        importance = message.get("importance", 1.0)
        weight = base_weight * importance
        message["weight"] = max(weight, self.min_weight)
        
        return True

    def get_weighted_messages(self, min_weight: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get messages with weights above the minimum threshold."""
        self._update_weights()
        
        if min_weight is None:
            min_weight = self.min_weight
        
        return [
            msg for msg in self.messages
            if msg["weight"] >= min_weight
        ]

    def get_recent_messages(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get messages from the last N hours."""
        self._update_weights()
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            msg for msg in self.messages
            if datetime.fromisoformat(msg["timestamp"]) >= cutoff_time
        ]

    def get_weighted_context(self, min_weight: Optional[float] = None) -> str:
        """Get context from messages weighted by recency."""
        weighted_messages = self.get_weighted_messages(min_weight)
        
        # Sort by weight in descending order
        weighted_messages.sort(key=lambda x: x["weight"], reverse=True)
        
        # Format context
        context = []
        for msg in weighted_messages:
            context.append(
                f"{msg['role']} (weight: {msg['weight']:.2f}, importance: {msg.get('importance', 1.0):.2f}): {msg['content']}"
            )
        
        return "\n".join(context)

    def get_average_weight(self) -> float:
        """Get the average weight of all messages."""
        self._update_weights()
        if not self.messages:
            return 0.0
        
        return sum(msg["weight"] for msg in self.messages) / len(self.messages)

    def get_message_count_by_weight(self, weight_threshold: float) -> int:
        """Get count of messages with weight above threshold."""
        self._update_weights()
        return sum(1 for msg in self.messages if msg["weight"] >= weight_threshold)

    def set_message_importance(self, message_index: int, importance: float) -> None:
        """Set the importance of a specific message."""
        if 0 <= message_index < len(self.messages):
            self.messages[message_index]["importance"] = max(0.0, min(1.0, importance))
            self._update_weights()
            self.save()

    def get_weight_distribution(self) -> Dict[str, float]:
        """Get distribution of message weights."""
        self._update_weights()
        if not self.messages:
            return {}
        
        weights = [msg["weight"] for msg in self.messages]
        return {
            "min": min(weights),
            "max": max(weights),
            "mean": sum(weights) / len(weights),
            "median": sorted(weights)[len(weights) // 2],
            "std_dev": math.sqrt(sum((w - sum(weights)/len(weights))**2 for w in weights) / len(weights))
        }

    def get_time_based_stats(self) -> Dict[str, Any]:
        """Get statistics about message timing."""
        if not self.messages:
            return {}
        
        timestamps = [datetime.fromisoformat(msg["timestamp"]) for msg in self.messages]
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                     for i in range(len(timestamps)-1)]
        
        return {
            "total_messages": len(self.messages),
            "time_span": (timestamps[-1] - timestamps[0]).total_seconds(),
            "avg_time_between_messages": sum(time_diffs) / len(time_diffs) if time_diffs else 0,
            "message_frequency": len(self.messages) / ((timestamps[-1] - timestamps[0]).total_seconds() / 3600)
            if len(timestamps) > 1 else 0
        }

    def get_importance_distribution(self) -> Dict[str, float]:
        """Get distribution of message importance scores."""
        if not self.messages:
            return {}
        
        importances = [msg.get("importance", 1.0) for msg in self.messages]
        return {
            "min": min(importances),
            "max": max(importances),
            "mean": sum(importances) / len(importances),
            "median": sorted(importances)[len(importances) // 2]
        } 