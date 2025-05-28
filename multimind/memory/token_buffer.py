"""
Token buffer memory implementation that manages messages based on token count.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path
from ..models.base import BaseLLM
from .base import BaseMemory

class TokenBufferMemory(BaseMemory):
    """Memory that manages messages based on token count."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_tokens: int = 2000,
        token_buffer: int = 100,  # Buffer to prevent exceeding max_tokens
        compression_threshold: int = 1500,  # Threshold for compression
        compression_ratio: float = 0.5,  # Target compression ratio
        min_tokens_per_message: int = 10  # Minimum tokens to keep per message
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_tokens = max_tokens
        self.token_buffer = token_buffer
        self.compression_threshold = compression_threshold
        self.compression_ratio = compression_ratio
        self.min_tokens_per_message = min_tokens_per_message
        self.messages: List[Dict[str, Any]] = []
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message and manage token count."""
        # Get token count for the new message
        token_count = await self._count_tokens(message["content"])
        
        message_with_metadata = {
            **message,
            "timestamp": datetime.now().isoformat(),
            "token_count": token_count,
            "original_content": message["content"]  # Keep original for potential recompression
        }
        
        # Add message and trim if needed
        self.messages.append(message_with_metadata)
        await self._manage_tokens()
        await self.save()

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages."""
        return self.messages

    async def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        await self.save()

    async def save(self) -> None:
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

    async def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the LLM."""
        try:
            return await self.llm.count_tokens(text)
        except Exception as e:
            print(f"Error counting tokens: {e}")
            # Fallback to rough estimation (4 chars per token)
            return len(text) // 4

    async def _manage_tokens(self) -> None:
        """Manage token count through trimming and compression."""
        total_tokens = self.get_current_token_count()
        
        if total_tokens > self.max_tokens:
            # First try compression
            if total_tokens > self.compression_threshold:
                await self._compress_messages()
                total_tokens = self.get_current_token_count()
            
            # If still over limit, trim messages
            if total_tokens > (self.max_tokens - self.token_buffer):
                await self._trim_to_token_limit()

    async def _compress_messages(self) -> None:
        """Compress messages to reduce token count."""
        if not self.messages:
            return
        
        # Group messages by role
        messages_by_role: Dict[str, List[Dict[str, Any]]] = {}
        for msg in self.messages:
            role = msg["role"]
            if role not in messages_by_role:
                messages_by_role[role] = []
            messages_by_role[role].append(msg)
        
        # Compress each role's messages
        compressed_messages = []
        for role, msgs in messages_by_role.items():
            if len(msgs) > 1:
                # Combine messages of the same role
                combined_content = "\n".join(msg["content"] for msg in msgs)
                target_tokens = int(sum(msg["token_count"] for msg in msgs) * self.compression_ratio)
                
                # Compress content
                compressed_content = await self._compress_content(combined_content, target_tokens)
                token_count = await self._count_tokens(compressed_content)
                
                compressed_messages.append({
                    "role": role,
                    "content": compressed_content,
                    "token_count": token_count,
                    "timestamp": datetime.now().isoformat(),
                    "original_content": combined_content,
                    "is_compressed": True
                })
            else:
                compressed_messages.extend(msgs)
        
        self.messages = compressed_messages

    async def _compress_content(self, content: str, target_tokens: int) -> str:
        """Compress content to target token count."""
        try:
            prompt = f"""
            Summarize the following content to approximately {target_tokens} tokens while preserving key information:
            {content}
            """
            return await self.llm.generate(prompt)
        except Exception as e:
            print(f"Error compressing content: {e}")
            return content

    async def _trim_to_token_limit(self) -> None:
        """Trim messages to stay within token limit."""
        total_tokens = self.get_current_token_count()
        
        while total_tokens > (self.max_tokens - self.token_buffer) and self.messages:
            # Remove oldest message
            removed_msg = self.messages.pop(0)
            total_tokens -= removed_msg["token_count"]

    def get_current_token_count(self) -> int:
        """Get current total token count."""
        return sum(msg["token_count"] for msg in self.messages)

    def get_remaining_tokens(self) -> int:
        """Get remaining token capacity."""
        return self.max_tokens - self.get_current_token_count()

    async def get_messages_within_token_limit(self, token_limit: int) -> List[Dict[str, Any]]:
        """Get messages that fit within a token limit."""
        result = []
        current_tokens = 0
        
        for msg in reversed(self.messages):  # Start from newest
            if current_tokens + msg["token_count"] <= token_limit:
                result.insert(0, msg)  # Add to beginning to maintain order
                current_tokens += msg["token_count"]
            else:
                break
        
        return result

    async def get_token_efficient_context(self, max_tokens: Optional[int] = None) -> str:
        """Get context that fits within token limit."""
        if max_tokens is None:
            max_tokens = self.max_tokens - self.token_buffer
        
        messages = await self.get_messages_within_token_limit(max_tokens)
        
        # Format context
        context = []
        for msg in messages:
            context.append(f"{msg['role']}: {msg['content']}")
        
        return "\n".join(context)

    def get_message_count_by_tokens(self, token_threshold: int) -> int:
        """Get count of messages with token count above threshold."""
        return sum(1 for msg in self.messages if msg["token_count"] >= token_threshold)

    def get_average_tokens_per_message(self) -> float:
        """Get average tokens per message."""
        if not self.messages:
            return 0.0
        
        return self.get_current_token_count() / len(self.messages)

    async def get_token_stats(self) -> Dict[str, Any]:
        """Get statistics about token usage."""
        if not self.messages:
            return {}
        
        token_counts = [msg["token_count"] for msg in self.messages]
        compressed_count = sum(1 for msg in self.messages if msg.get("is_compressed", False))
        
        return {
            "total_tokens": sum(token_counts),
            "message_count": len(self.messages),
            "average_tokens": sum(token_counts) / len(token_counts),
            "max_tokens": max(token_counts),
            "min_tokens": min(token_counts),
            "compressed_messages": compressed_count,
            "compression_ratio": compressed_count / len(self.messages) if self.messages else 0
        }

    async def get_role_token_distribution(self) -> Dict[str, int]:
        """Get token distribution by role."""
        distribution = {}
        for msg in self.messages:
            role = msg["role"]
            distribution[role] = distribution.get(role, 0) + msg["token_count"]
        return distribution

    async def get_compression_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for potential message compression."""
        suggestions = []
        
        # Group messages by role
        messages_by_role: Dict[str, List[Dict[str, Any]]] = {}
        for msg in self.messages:
            role = msg["role"]
            if role not in messages_by_role:
                messages_by_role[role] = []
            messages_by_role[role].append(msg)
        
        # Analyze each role's messages
        for role, msgs in messages_by_role.items():
            if len(msgs) > 1:
                total_tokens = sum(msg["token_count"] for msg in msgs)
                if total_tokens > self.compression_threshold:
                    suggestions.append({
                        "role": role,
                        "message_count": len(msgs),
                        "total_tokens": total_tokens,
                        "potential_savings": int(total_tokens * (1 - self.compression_ratio))
                    })
        
        return suggestions 