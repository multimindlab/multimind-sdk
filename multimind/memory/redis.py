"""
Redis-based memory implementation.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
import redis
from .base import BaseMemory

class RedisMemory(BaseMemory):
    """Memory that uses Redis for storage."""

    def __init__(
        self,
        redis_url: str,
        memory_key: str = "chat_history",
        ttl: Optional[int] = None
    ):
        super().__init__(memory_key)
        self.redis_client = redis.from_url(redis_url)
        self.ttl = ttl  # Time to live in seconds

    def add_message(self, message: Dict[str, str]) -> None:
        """Add message to Redis."""
        message_with_timestamp = {
            **message,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to Redis list
        self.redis_client.rpush(
            self.memory_key,
            json.dumps(message_with_timestamp)
        )
        
        # Set TTL if specified
        if self.ttl:
            self.redis_client.expire(self.memory_key, self.ttl)

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from Redis."""
        messages = self.redis_client.lrange(self.memory_key, 0, -1)
        return [json.loads(msg) for msg in messages]

    def clear(self) -> None:
        """Clear all messages from Redis."""
        self.redis_client.delete(self.memory_key)

    def save(self) -> None:
        """Save is handled automatically by Redis."""
        pass

    def load(self) -> None:
        """Load is handled automatically by Redis."""
        pass

    def get_message_count(self) -> int:
        """Get the number of messages in memory."""
        return self.redis_client.llen(self.memory_key)

    def get_messages_since(self, timestamp: datetime) -> List[Dict[str, str]]:
        """Get messages since a specific timestamp."""
        all_messages = self.get_messages()
        return [
            msg for msg in all_messages
            if datetime.fromisoformat(msg["timestamp"]) > timestamp
        ]

    def trim_messages(self, max_messages: int) -> None:
        """Trim the message list to a maximum size."""
        current_count = self.get_message_count()
        if current_count > max_messages:
            self.redis_client.ltrim(
                self.memory_key,
                current_count - max_messages,
                -1
            ) 