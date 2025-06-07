"""
Summary memory implementation for storing summarized conversations.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path
from .base import BaseMemory
from ..models.base import BaseLLM

class SummaryMemory(BaseMemory):
    """Memory that stores summarized versions of conversations."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_summaries: int = 100,
        summary_interval: int = 10,  # Number of messages between summaries
        strategy: str = "extractive",  # extractive, abstractive, or hybrid
        min_summary_length: int = 100,
        max_summary_length: int = 500,
        enable_metadata: bool = True,
        enable_compression: bool = True,
        compression_threshold: float = 0.8,
        enable_backup: bool = True,
        backup_interval: int = 3600,  # 1 hour
        max_backups: int = 5
    ):
        """Initialize summary memory."""
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_summaries = max_summaries
        self.summary_interval = summary_interval
        self.strategy = strategy
        self.min_summary_length = min_summary_length
        self.max_summary_length = max_summary_length
        self.enable_metadata = enable_metadata
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.enable_backup = enable_backup
        self.backup_interval = backup_interval
        self.max_backups = max_backups

        # Initialize storage
        self.summaries: List[Dict[str, Any]] = []
        self.summary_metadata: Dict[str, Dict[str, Any]] = {}
        self.message_count: int = 0
        self.last_summary: Optional[datetime] = None
        self.last_backup = datetime.now()
        self.backup_history: List[Dict[str, Any]] = []

        # Load if storage path exists
        if self.storage_path and self.storage_path.exists():
            self.load()

    async def add_messages(
        self,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add messages and generate summary if needed."""
        self.message_count += len(messages)

        # Check if we need to generate a summary
        if (
            self.message_count >= self.summary_interval or
            not self.summaries
        ):
            await self._generate_summary(messages, metadata)
            self.message_count = 0
            self.last_summary = datetime.now()

    async def _generate_summary(
        self,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Generate a summary of the messages."""
        if not messages:
            return

        # Prepare messages for summarization
        message_texts = [
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages
        ]
        combined_text = "\n".join(message_texts)

        # Generate summary based on strategy
        if self.strategy == "extractive":
            summary = await self._extractive_summarize(combined_text)
        elif self.strategy == "abstractive":
            summary = await self._abstractive_summarize(combined_text)
        else:  # hybrid
            summary = await self._hybrid_summarize(combined_text)

        # Create summary entry
        summary_entry = {
            "content": summary,
            "timestamp": datetime.now().isoformat(),
            "message_count": len(messages),
            "metadata": metadata or {}
        }

        # Add to summaries
        self.summaries.append(summary_entry)
        if self.enable_metadata:
            self.summary_metadata[str(len(self.summaries) - 1)] = metadata or {}

        # Trim if needed
        if len(self.summaries) > self.max_summaries:
            self.summaries = self.summaries[-self.max_summaries:]
            if self.enable_metadata:
                new_metadata = {}
                for i in range(len(self.summaries)):
                    new_metadata[str(i)] = self.summary_metadata.get(str(i + len(self.summaries) - self.max_summaries), {})
                self.summary_metadata = new_metadata

        # Check if compression needed
        if self.enable_compression and len(self.summaries) > self.max_summaries * self.compression_threshold:
            await self._compress_summaries()

        # Check if backup needed
        if self.enable_backup and (datetime.now() - self.last_backup).total_seconds() >= self.backup_interval:
            await self._backup()

    async def _extractive_summarize(self, text: str) -> str:
        """Generate extractive summary."""
        # This is a placeholder for actual extractive summarization
        # In practice, you would use more sophisticated methods
        # For example, using an LLM to select key sentences
        prompt = f"""Extract the key points from this conversation:

{text}

Key points:"""
        response = await self.llm.generate(prompt)
        return response

    async def _abstractive_summarize(self, text: str) -> str:
        """Generate abstractive summary."""
        # This is a placeholder for actual abstractive summarization
        # In practice, you would use more sophisticated methods
        # For example, using an LLM to generate a concise summary
        prompt = f"""Summarize this conversation in a concise way:

{text}

Summary:"""
        response = await self.llm.generate(prompt)
        return response

    async def _hybrid_summarize(self, text: str) -> str:
        """Generate hybrid summary."""
        # This is a placeholder for actual hybrid summarization
        # In practice, you would use more sophisticated methods
        # For example, combining extractive and abstractive approaches
        extractive = await self._extractive_summarize(text)
        abstractive = await self._abstractive_summarize(text)
        return f"{extractive}\n\n{abstractive}"

    async def _compress_summaries(self) -> None:
        """Compress summaries to reduce storage."""
        if not self.enable_compression or not self.summaries:
            return

        # This is a placeholder for actual compression logic
        # In practice, you would use more sophisticated compression
        # For example, using an LLM to combine similar summaries
        pass

    async def _backup(self) -> None:
        """Create a backup of the current summary state."""
        if not self.enable_backup:
            return

        backup = {
            "timestamp": datetime.now().isoformat(),
            "summaries": self.summaries,
            "summary_metadata": self.summary_metadata,
            "message_count": self.message_count,
            "last_summary": self.last_summary.isoformat() if self.last_summary else None
        }

        self.backup_history.append(backup)
        self.last_backup = datetime.now()

        # Trim backup history if needed
        if len(self.backup_history) > self.max_backups:
            self.backup_history = self.backup_history[-self.max_backups:]

        # Save to disk if storage path exists
        if self.storage_path:
            self.save()

    def get_summaries(self) -> List[Dict[str, Any]]:
        """Get all summaries."""
        return self.summaries

    def get_summaries_with_metadata(self) -> List[Dict[str, Any]]:
        """Get summaries with their metadata."""
        return [
            {
                "summary": summary,
                "metadata": self.summary_metadata.get(str(i), {})
            }
            for i, summary in enumerate(self.summaries)
        ]

    def clear(self) -> None:
        """Clear all summaries."""
        self.summaries = []
        self.summary_metadata = {}
        self.message_count = 0
        self.last_summary = None
        if self.storage_path and self.storage_path.exists():
            self.storage_path.unlink()

    def save(self) -> None:
        """Save summaries to persistent storage."""
        if not self.storage_path:
            return

        data = {
            "summaries": self.summaries,
            "summary_metadata": self.summary_metadata,
            "message_count": self.message_count,
            "last_summary": self.last_summary.isoformat() if self.last_summary else None,
            "last_backup": self.last_backup.isoformat(),
            "backup_history": self.backup_history
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f)

    def load(self) -> None:
        """Load summaries from persistent storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            self.summaries = data["summaries"]
            self.summary_metadata = data["summary_metadata"]
            self.message_count = data["message_count"]
            self.last_summary = datetime.fromisoformat(data["last_summary"]) if data["last_summary"] else None
            self.last_backup = datetime.fromisoformat(data["last_backup"])
            self.backup_history = data["backup_history"]
        except Exception as e:
            print(f"Error loading summaries: {e}")
            self.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_summaries": len(self.summaries),
            "max_summaries": self.max_summaries,
            "summary_interval": self.summary_interval,
            "strategy": self.strategy,
            "message_count": self.message_count,
            "last_summary": self.last_summary.isoformat() if self.last_summary else None,
            "enable_metadata": self.enable_metadata,
            "enable_compression": self.enable_compression,
            "enable_backup": self.enable_backup,
            "last_backup": self.last_backup.isoformat(),
            "backup_count": len(self.backup_history)
        } 