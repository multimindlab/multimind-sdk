"""
Conversation summary memory implementation.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
from pathlib import Path
from ..models.base import BaseLLM
from .base import BaseMemory

class ConversationSummaryMemory(BaseMemory):
    """Memory that maintains a summary of the conversation."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        max_token_limit: int = 2000,
        storage_path: Optional[str] = None
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.storage_path = Path(storage_path) if storage_path else None
        self.messages: List[Dict[str, str]] = []
        self.summary: str = ""
        self.load()

    def add_message(self, message: Dict[str, str]) -> None:
        """Add a message and update the summary."""
        self.messages.append({
            **message,
            "timestamp": datetime.now().isoformat()
        })
        self._update_summary()
        self.save()

    def get_messages(self) -> List[Dict[str, str]]:
        """Get the conversation summary and recent messages."""
        return [{"role": "system", "content": self.summary}] + self.messages[-5:]

    def clear(self) -> None:
        """Clear all messages and summary."""
        self.messages.clear()
        self.summary = ""
        self.save()

    def save(self) -> None:
        """Save messages and summary to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "messages": self.messages,
                    "summary": self.summary
                }, f)

    def load(self) -> None:
        """Load messages and summary from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.messages = data.get("messages", [])
                self.summary = data.get("summary", "")

    async def _update_summary(self) -> None:
        """Update the conversation summary using the LLM."""
        if not self.messages:
            return

        # Create a prompt for summarization
        prompt = f"""Please provide a concise summary of the following conversation:

{self._format_messages_for_summary()}

Summary:"""

        # Get summary from LLM
        try:
            new_summary = await self.llm.generate(prompt)
            self.summary = new_summary
        except Exception as e:
            print(f"Error updating summary: {e}")

    def _format_messages_for_summary(self) -> str:
        """Format messages for summarization."""
        return "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.messages[-10:]  # Use last 10 messages for summary
        ]) 