"""
Read-only memory wrapper that prevents modifications to the underlying memory.
"""

from typing import List, Dict, Any, Optional
from .base import BaseMemory

class ReadOnlyMemory(BaseMemory):
    """Memory wrapper that prevents modifications to the underlying memory."""

    def __init__(self, memory: BaseMemory, **kwargs):
        """Initialize read-only memory wrapper."""
        super().__init__(**kwargs)
        self._memory = memory

    async def add_message(
        self,
        message: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Raise error - read-only memory cannot be modified."""
        raise RuntimeError("Cannot modify read-only memory")

    async def get_messages(
        self,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get messages from underlying memory."""
        return await self._memory.get_messages(**kwargs)

    async def clear(self) -> None:
        """Raise error - read-only memory cannot be cleared."""
        raise RuntimeError("Cannot modify read-only memory")

    async def get_message_count(self) -> int:
        """Get message count from underlying memory."""
        return await self._memory.get_message_count()

    async def get_oldest_message(self) -> Optional[Dict[str, Any]]:
        """Get oldest message from underlying memory."""
        return await self._memory.get_oldest_message()

    async def get_newest_message(self) -> Optional[Dict[str, Any]]:
        """Get newest message from underlying memory."""
        return await self._memory.get_newest_message()

    async def get_messages_by_role(self, role: str) -> List[Dict[str, Any]]:
        """Get messages by role from underlying memory."""
        return await self._memory.get_messages_by_role(role)

    async def get_messages_in_timeframe(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get messages in timeframe from underlying memory."""
        return await self._memory.get_messages_in_timeframe(start_time, end_time)

    @property
    def memory(self) -> BaseMemory:
        """Get the underlying memory instance."""
        return self._memory 