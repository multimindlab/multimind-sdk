"""
Combined memory implementation that uses multiple memory types.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from .base import BaseMemory

class CombinedMemory(BaseMemory):
    """Memory that combines multiple memory types."""

    def __init__(
        self,
        memories: List[BaseMemory],
        memory_key: str = "chat_history"
    ):
        super().__init__(memory_key)
        self.memories = memories

    def add_message(self, message: Dict[str, str]) -> None:
        """Add message to all memory types."""
        for memory in self.memories:
            memory.add_message(message)

    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages from all memory types."""
        all_messages = []
        for memory in self.memories:
            all_messages.extend(memory.get_messages())
        return all_messages

    def clear(self) -> None:
        """Clear all memory types."""
        for memory in self.memories:
            memory.clear()

    def save(self) -> None:
        """Save all memory types."""
        for memory in self.memories:
            memory.save()

    def load(self) -> None:
        """Load all memory types."""
        for memory in self.memories:
            memory.load()

    def get_memory(self, memory_type: type) -> Optional[BaseMemory]:
        """Get a specific memory type instance."""
        for memory in self.memories:
            if isinstance(memory, memory_type):
                return memory
        return None 