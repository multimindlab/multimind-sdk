"""
Conversation buffer memory implementation.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
from pathlib import Path
from .base import BaseMemory

class ConversationBufferMemory(BaseMemory):
    """Memory that stores all messages in a buffer."""

    def __init__(
        self,
        memory_key: str = "chat_history",
        return_messages: bool = False,
        storage_path: Optional[str] = None
    ):
        super().__init__(memory_key)
        self.return_messages = return_messages
        self.storage_path = Path(storage_path) if storage_path else None
        self.messages: List[Dict[str, str]] = []
        self.load()

    def add_message(self, message: Dict[str, str]) -> None:
        """Add a message to the buffer."""
        self.messages.append({
            **message,
            "timestamp": datetime.now().isoformat()
        })
        self.save()

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from the buffer."""
        if self.return_messages:
            return self.messages
        return [f"{msg['role']}: {msg['content']}" for msg in self.messages]

    def clear(self) -> None:
        """Clear all messages from the buffer."""
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