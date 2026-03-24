"""
Redis-based memory implementation.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
import redis.asyncio as redis  # type: ignore[import-not-found]
from redis import exceptions as redis_exceptions
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
        try:
            # decode_responses=True makes Redis return str (not bytes) for reads like LRANGE.
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
        except redis_exceptions.RedisError as e:
            raise RuntimeError(f"Redis connection error: {e}") from e
        self.ttl = ttl  # Time to live in seconds
        self._validated = False

    async def _redis_call(self, fn_name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a Redis call with consistent connection error handling."""
        try:
            # Lazy connection validation (fail-fast on first real use).
            if not self._validated:
                await self.redis_client.ping()
                self._validated = True

            fn = getattr(self.redis_client, fn_name)
            return await fn(*args, **kwargs)
        except AttributeError as e:
            raise RuntimeError(f"Redis client has no method '{fn_name}'") from e
        except redis_exceptions.RedisError as e:
            raise RuntimeError(f"Redis operation failed ({fn_name}): {e}") from e

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message to Redis."""
        message_with_timestamp = {
            **message,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to Redis list
        await self._redis_call(
            "rpush",
            self.memory_key,
            json.dumps(message_with_timestamp)
        )
        
        # Set TTL if specified
        if self.ttl:
            await self._redis_call("expire", self.memory_key, self.ttl)

    async def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from Redis."""
        messages = await self._redis_call("lrange", self.memory_key, 0, -1)
        return [json.loads(msg) for msg in messages]

    async def clear(self) -> None:
        """Clear all messages from Redis."""
        await self._redis_call("delete", self.memory_key)

    async def save(self) -> None:
        """Save is handled automatically by Redis."""
        pass

    async def load(self) -> None:
        """Load is handled automatically by Redis."""
        pass

    async def get_message_count(self) -> int:
        """Get the number of messages in memory."""
        return int(await self._redis_call("llen", self.memory_key))

    async def get_messages_since(self, timestamp: datetime) -> List[Dict[str, str]]:
        """Get messages since a specific timestamp."""
        all_messages = await self._redis_call("lrange", self.memory_key, 0, -1)
        all_messages = [json.loads(msg) for msg in all_messages]
        return [
            msg for msg in all_messages
            if datetime.fromisoformat(msg["timestamp"]) > timestamp
        ]

    async def trim_messages(self, max_messages: int) -> None:
        """Trim the message list to a maximum size."""
        current_count = await self.get_message_count()
        if current_count > max_messages:
            await self._redis_call(
                "ltrim",
                self.memory_key,
                current_count - max_messages,
                -1
            ) 