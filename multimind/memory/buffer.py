"""
Buffer memory implementation for managing recent context.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path
from .base import BaseMemory

class BufferMemory(BaseMemory):
    """Memory that maintains a buffer of recent messages with token management."""

    def __init__(
        self,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_tokens: int = 2000,
        max_messages: Optional[int] = None,
        tokenizer: Optional[Any] = None,
        strategy: str = "sliding",  # sliding, fifo, or lru
        enable_token_tracking: bool = True,
        enable_metadata: bool = True,
        enable_compression: bool = False,
        compression_threshold: float = 0.8,
        enable_backup: bool = True,
        backup_interval: int = 3600,  # 1 hour
        max_backups: int = 5
    ):
        """Initialize buffer memory."""
        super().__init__(memory_key)
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.enable_token_tracking = enable_token_tracking
        self.enable_metadata = enable_metadata
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.enable_backup = enable_backup
        self.backup_interval = backup_interval
        self.max_backups = max_backups

        # Initialize storage
        self.messages: List[Dict[str, Any]] = []
        self.message_tokens: List[int] = []  # Token count per message
        self.total_tokens: int = 0
        self.metadata: Dict[str, Any] = {}
        self.last_backup = datetime.now()
        self.backup_history: List[Dict[str, Any]] = []

        # Load if storage path exists
        if self.storage_path and self.storage_path.exists():
            self.load()

    async def add_message(
        self,
        message: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a message to the buffer."""
        # Calculate tokens if tracking enabled
        tokens = 0
        if self.enable_token_tracking and self.tokenizer:
            tokens = len(self.tokenizer.encode(message.get("content", "")))
        elif self.enable_token_tracking:
            # Simple token estimation if no tokenizer
            tokens = len(message.get("content", "").split())

        # Check if we need to make space
        while (
            (self.max_tokens and self.total_tokens + tokens > self.max_tokens) or
            (self.max_messages and len(self.messages) >= self.max_messages)
        ):
            if not self.messages:
                break
            self._remove_oldest()

        # Add message
        self.messages.append(message)
        self.message_tokens.append(tokens)
        self.total_tokens += tokens

        # Add metadata if enabled
        if self.enable_metadata and metadata:
            self.metadata[str(len(self.messages) - 1)] = metadata

        # Check if compression needed
        if self.enable_compression and self.total_tokens > self.max_tokens * self.compression_threshold:
            await self._compress_messages()

        # Check if backup needed
        if self.enable_backup and (datetime.now() - self.last_backup).total_seconds() >= self.backup_interval:
            await self._backup()

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from the buffer."""
        return self.messages

    def get_messages_with_metadata(self) -> List[Dict[str, Any]]:
        """Get messages with their metadata."""
        return [
            {
                "message": msg,
                "metadata": self.metadata.get(str(i), {})
            }
            for i, msg in enumerate(self.messages)
        ]

    def clear(self) -> None:
        """Clear all messages from the buffer."""
        self.messages = []
        self.message_tokens = []
        self.total_tokens = 0
        self.metadata = {}
        if self.storage_path and self.storage_path.exists():
            self.storage_path.unlink()

    def save(self) -> None:
        """Save buffer to persistent storage."""
        if not self.storage_path:
            return

        data = {
            "messages": self.messages,
            "message_tokens": self.message_tokens,
            "total_tokens": self.total_tokens,
            "metadata": self.metadata,
            "last_backup": self.last_backup.isoformat(),
            "backup_history": self.backup_history
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f)

    def load(self) -> None:
        """Load buffer from persistent storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            self.messages = data["messages"]
            self.message_tokens = data["message_tokens"]
            self.total_tokens = data["total_tokens"]
            self.metadata = data["metadata"]
            self.last_backup = datetime.fromisoformat(data["last_backup"])
            self.backup_history = data["backup_history"]
        except Exception as e:
            print(f"Error loading buffer: {e}")
            self.clear()

    def _remove_oldest(self) -> None:
        """Remove the oldest message based on strategy."""
        if not self.messages:
            return

        if self.strategy == "fifo":
            # Remove first message
            self.total_tokens -= self.message_tokens[0]
            self.messages.pop(0)
            self.message_tokens.pop(0)
            if self.enable_metadata:
                self.metadata.pop("0", None)
                # Shift metadata indices
                new_metadata = {}
                for i in range(len(self.messages)):
                    new_metadata[str(i)] = self.metadata.get(str(i + 1), {})
                self.metadata = new_metadata

        elif self.strategy == "lru":
            # Remove least recently used message
            # This is a simplified implementation
            # In practice, you would track access times
            self._remove_oldest()

        else:  # sliding
            # Remove messages from start until we have space
            while self.messages and (
                (self.max_tokens and self.total_tokens > self.max_tokens) or
                (self.max_messages and len(self.messages) >= self.max_messages)
            ):
                self.total_tokens -= self.message_tokens[0]
                self.messages.pop(0)
                self.message_tokens.pop(0)
                if self.enable_metadata:
                    self.metadata.pop("0", None)
                    # Shift metadata indices
                    new_metadata = {}
                    for i in range(len(self.messages)):
                        new_metadata[str(i)] = self.metadata.get(str(i + 1), {})
                    self.metadata = new_metadata

    def set_compression_strategy(self, strategy: str, llm: Optional[Any] = None):
        """Set the compression strategy (llm, truncate, concat) and optional LLM."""
        self.compression_strategy = strategy
        self.compression_llm = llm

    async def _compress_messages(self) -> None:
        """Compress messages to reduce token usage (adaptive/LLM-based)."""
        if not self.enable_compression or not self.messages:
            return
        n = len(self.messages)
        if n < 2:
            return
        half = n // 2
        to_compress = self.messages[:half]
        summary_content = None
        method_used = self.compression_strategy if hasattr(self, 'compression_strategy') else 'concat'
        if hasattr(self, 'compression_strategy') and self.compression_strategy == 'llm' and hasattr(self, 'compression_llm') and self.compression_llm:
            # Use LLM to summarize
            prompt = "Summarize the following conversation:\n" + "\n".join([msg.get("content", "") for msg in to_compress])
            try:
                summary_content = await self.compression_llm.generate(prompt)
                method_used = 'llm'
            except Exception:
                summary_content = " ".join([msg.get("content", "") for msg in to_compress])[:256] + "..."
                method_used = 'concat_fallback'
        elif hasattr(self, 'compression_strategy') and self.compression_strategy == 'truncate':
            summary_content = " ".join([msg.get("content", "") for msg in to_compress])[:256] + "..."
            method_used = 'truncate'
        else:
            summary_content = " ".join([msg.get("content", "") for msg in to_compress])[:256] + "..."
            method_used = 'concat'
        summary_message = {"role": "system", "content": f"Summary: {summary_content}", "compression_method": method_used}
        # Remove the oldest half and insert the summary at the start
        self.messages = [summary_message] + self.messages[half:]
        self.message_tokens = [len(summary_content.split())] + self.message_tokens[half:]
        self.total_tokens = sum(self.message_tokens)

    async def _backup(self) -> None:
        """Create a backup of the current buffer state."""
        if not self.enable_backup:
            return

        backup = {
            "timestamp": datetime.now().isoformat(),
            "messages": self.messages,
            "message_tokens": self.message_tokens,
            "total_tokens": self.total_tokens,
            "metadata": self.metadata
        }

        self.backup_history.append(backup)
        self.last_backup = datetime.now()

        # Trim backup history if needed
        if len(self.backup_history) > self.max_backups:
            self.backup_history = self.backup_history[-self.max_backups:]

        # Save to disk if storage path exists
        if self.storage_path:
            self.save()

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "total_messages": len(self.messages),
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "max_messages": self.max_messages,
            "strategy": self.strategy,
            "enable_token_tracking": self.enable_token_tracking,
            "enable_metadata": self.enable_metadata,
            "enable_compression": self.enable_compression,
            "enable_backup": self.enable_backup,
            "last_backup": self.last_backup.isoformat(),
            "backup_count": len(self.backup_history)
        } 