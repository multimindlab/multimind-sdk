"""
Base memory class for all memory implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime

class BaseMemory(ABC):
    """Abstract base class for all memory implementations."""

    def __init__(self, memory_key: str = "chat_history"):
        self.memory_key = memory_key
        self.created_at = datetime.now()

    @abstractmethod
    async def add_message(self, message: Dict[str, str]) -> None:
        """Add a message to memory."""
        pass

    @abstractmethod
    async def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from memory."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all messages from memory."""
        pass

    @abstractmethod
    async def save(self) -> None:
        """Save memory to persistent storage."""
        pass

    @abstractmethod
    async def load(self) -> None:
        """Load memory from persistent storage."""
        pass 