"""
Sliding window buffer memory implementation that maintains a fixed-size window of recent messages.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from .buffer import BufferMemory

class BufferWindowMemory(BufferMemory):
    """Memory that maintains a sliding window of recent messages."""

    def __init__(
        self,
        window_size: int = 10,
        window_type: str = "count",  # count, time, or tokens
        window_value: Optional[Any] = None,  # count, timedelta, or token count
        **kwargs
    ):
        """Initialize buffer window memory."""
        super().__init__(**kwargs)
        self.window_size = window_size
        self.window_type = window_type
        self.window_value = window_value or (
            timedelta(hours=1) if window_type == "time"
            else 1000 if window_type == "tokens"
            else 10
        )

    async def add_message(
        self,
        message: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a message and maintain window."""
        await super().add_message(message, metadata)
        await self._maintain_window()

    async def _maintain_window(self) -> None:
        """Maintain the sliding window based on window type."""
        if self.window_type == "count":
            await self._maintain_count_window()
        elif self.window_type == "time":
            await self._maintain_time_window()
        else:  # tokens
            await self._maintain_token_window()

    async def _maintain_count_window(self) -> None:
        """Maintain window based on message count."""
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]

    async def _maintain_time_window(self) -> None:
        """Maintain window based on time."""
        cutoff_time = datetime.now() - self.window_value
        self.messages = [
            m for m in self.messages
            if m["timestamp"] >= cutoff_time
        ]

    async def _maintain_token_window(self) -> None:
        """Maintain window based on token count."""
        from .token_buffer import TokenBufferMemory
        token_memory = TokenBufferMemory(max_tokens=self.window_value)
        
        # Add messages to token memory
        for msg in self.messages:
            await token_memory.add_message(msg["message"], msg["metadata"])
        
        # Get messages that fit within token limit
        self.messages = [
            {
                "message": m["message"],
                "metadata": m["metadata"],
                "timestamp": m["timestamp"]
            }
            for m in token_memory.messages
        ]

    async def get_window_stats(self) -> Dict[str, Any]:
        """Get statistics about the current window."""
        if not self.messages:
            return {
                "window_type": self.window_type,
                "window_value": self.window_value,
                "message_count": 0,
                "window_usage": 0.0
            }
            
        if self.window_type == "count":
            usage = len(self.messages) / self.window_size
        elif self.window_type == "time":
            oldest = self.messages[0]["timestamp"]
            window_span = datetime.now() - oldest
            usage = window_span / self.window_value
        else:  # tokens
            from .token_buffer import TokenBufferMemory
            token_memory = TokenBufferMemory(max_tokens=self.window_value)
            for msg in self.messages:
                await token_memory.add_message(msg["message"], msg["metadata"])
            usage = token_memory.total_tokens / self.window_value
            
        return {
            "window_type": self.window_type,
            "window_value": self.window_value,
            "message_count": len(self.messages),
            "window_usage": min(1.0, usage),
            "oldest_message": self.messages[0]["timestamp"],
            "newest_message": self.messages[-1]["timestamp"]
        } 