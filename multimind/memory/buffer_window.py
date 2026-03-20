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
        # Validate window_type early
        if window_type not in {"count", "time", "tokens"}:
            raise ValueError(f"Invalid window_type: {window_type}")

        self.window_size = window_size
        self.window_type = window_type
        # Normalize window_value so all logic uses a single field
        if window_type == "count":
            # For count-based windows, treat window_value as the max message count
            self.window_value = int(window_value or window_size)
        elif window_type == "time":
            self.window_value = window_value or timedelta(hours=1)
        else:  # tokens
            self.window_value = int(window_value or 1000)

    async def add_message(
        self,
        message: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a message and maintain window."""
        # Attach a timestamp field so windowing can operate on time.
        message_with_timestamp: Dict[str, Any] = {
            **message,
            "timestamp": datetime.now().isoformat(),
        }
        await super().add_message(message_with_timestamp, metadata)
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
        if len(self.messages) > self.window_value:
            # Trim oldest messages to keep at most window_value messages
            excess = len(self.messages) - self.window_value
            if excess > 0:
                # Drop from the front
                self.messages = self.messages[excess:]

    async def _maintain_time_window(self) -> None:
        """Maintain window based on time."""
        cutoff_time = datetime.now() - self.window_value
        self.messages = [
            m for m in self.messages
            if "timestamp" in m
            and datetime.fromisoformat(str(m["timestamp"])) >= cutoff_time
        ]

    async def _maintain_token_window(self) -> None:
        """Maintain window based on token count."""
        # Use existing token accounting from BufferMemory to trim in O(n)
        if not self.enable_token_tracking:
            return

        removed = 0
        # Drop oldest messages until we are within the token budget
        while self.total_tokens > self.window_value and self.messages:
            # Remove oldest message and its token count
            removed_tokens = self.message_tokens.pop(0)
            self.messages.pop(0)
            self.total_tokens -= removed_tokens
            removed += 1

        # Re-align metadata indices to the new message ordering, if metadata is used
        if self.enable_metadata and self.metadata:
            new_metadata: Dict[str, Any] = {}
            for new_idx in range(len(self.messages)):
                old_key = str(new_idx + removed)
                if old_key in self.metadata:
                    new_metadata[str(new_idx)] = self.metadata[old_key]
            self.metadata = new_metadata

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
            usage = len(self.messages) / max(1, self.window_value)
        elif self.window_type == "time":
            oldest_str = self.messages[0].get("timestamp")
            try:
                oldest = datetime.fromisoformat(str(oldest_str)) if oldest_str else datetime.now()
            except Exception:
                oldest = datetime.now()
            window_span = datetime.now() - oldest
            usage = window_span / self.window_value
        else:  # tokens
            # Use the existing token accounting from BufferMemory
            usage = (self.total_tokens / float(self.window_value)) if self.window_value else 0.0
            
        return {
            "window_type": self.window_type,
            "window_value": self.window_value,
            "message_count": len(self.messages),
            "window_usage": min(1.0, usage),
            "oldest_message": self.messages[0].get("timestamp"),
            "newest_message": self.messages[-1].get("timestamp"),
        } 