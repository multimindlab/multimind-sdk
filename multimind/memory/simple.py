"""
Simple memory implementation with basic functionality.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from .base import BaseMemory

class SimpleMemory(BaseMemory):
    """Simple memory implementation with basic functionality."""

    def __init__(
        self,
        max_messages: Optional[int] = None,
        **kwargs
    ):
        """Initialize simple memory."""
        super().__init__(**kwargs)
        self.max_messages = max_messages
        self.messages: List[Dict[str, Any]] = []

    async def add_message(
        self,
        message: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a message to memory."""
        # Add timestamp and metadata
        message_with_metadata = {
            "message": message,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        }
        
        # Add to memory
        self.messages.append(message_with_metadata)
        
        # Trim if needed
        if self.max_messages and len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    async def get_messages(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get messages from memory."""
        messages = self.messages[offset:]
        if limit:
            messages = messages[:limit]
        return [m["message"] for m in messages]

    async def clear(self) -> None:
        """Clear all messages."""
        self.messages = []

    async def get_message_count(self) -> int:
        """Get the number of messages."""
        return len(self.messages)

    async def get_oldest_message(self) -> Optional[Dict[str, Any]]:
        """Get the oldest message."""
        if not self.messages:
            return None
        return self.messages[0]["message"]

    async def get_newest_message(self) -> Optional[Dict[str, Any]]:
        """Get the newest message."""
        if not self.messages:
            return None
        return self.messages[-1]["message"]

    async def get_messages_by_role(self, role: str) -> List[Dict[str, Any]]:
        """Get messages by role."""
        return [
            m["message"] for m in self.messages
            if m["message"].get("role") == role
        ]

    async def get_messages_in_timeframe(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get messages within a timeframe."""
        return [
            m["message"] for m in self.messages
            if start_time <= m["timestamp"] <= end_time
        ]

    async def get_messages_with_metadata(
        self,
        metadata_key: str,
        metadata_value: Any
    ) -> List[Dict[str, Any]]:
        """Get messages with specific metadata."""
        return [
            m["message"] for m in self.messages
            if m["metadata"].get(metadata_key) == metadata_value
        ]

    async def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Get all message metadata."""
        return [m["metadata"] for m in self.messages]

    async def get_message_with_metadata(
        self,
        index: int
    ) -> Optional[Dict[str, Any]]:
        """Get a message with its metadata."""
        if not 0 <= index < len(self.messages):
            return None
        return self.messages[index] 