"""
Summary buffer memory implementation that maintains a buffer of messages with summaries.

Features:
- Supports advanced/contextual similarity metrics for relevance
- Supports adaptive thresholds for filtering/relevance
- Buffer strategies: sliding, fixed, dynamic (with adaptive/contextual relevance)
Usage:
    buffer = SummaryBufferMemory(...)
    buffer.set_similarity_func(custom_similarity)
    buffer.set_adaptive_threshold(AdaptiveThreshold(...))
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from .summary import SummaryMemory
from .buffer import BufferMemory

class SummaryBufferMemory(SummaryMemory):
    """Memory that maintains a buffer of messages with summaries."""

    def __init__(
        self,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        summary_interval: int = 10,  # messages
        summary_strategy: str = "extractive",  # extractive, abstractive, hybrid
        max_summaries: int = 5,
        buffer_strategy: str = "sliding",  # sliding, fixed, dynamic
        **kwargs
    ):
        """Initialize summary buffer memory."""
        super().__init__(
            max_messages=max_messages,
            summary_interval=summary_interval,
            summary_strategy=summary_strategy,
            max_summaries=max_summaries,
            **kwargs
        )
        
        # Buffer configuration
        self.max_tokens = max_tokens
        self.buffer_strategy = buffer_strategy
        
        # Buffer state
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_tokens = 0
        self.last_buffer_update = datetime.now()

    async def add_message(
        self,
        message: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a message to memory and buffer."""
        # Add to main memory
        await super().add_message(message, metadata)
        
        # Add to buffer
        await self._add_to_buffer(message, metadata)
        
        # Check if we should update buffer
        if (
            self.buffer_strategy == "dynamic" and
            datetime.now() - self.last_buffer_update >= timedelta(minutes=5)
        ):
            await self._update_buffer()

    async def _add_to_buffer(
        self,
        message: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add message to buffer."""
        # Calculate tokens
        content = message.get("content", "")
        tokens = len(self.tokenizer.encode(content)) if self.max_tokens else 0
        
        # Add to buffer
        self.buffer.append({
            "message": message,
            "metadata": metadata or {},
            "tokens": tokens,
            "timestamp": datetime.now()
        })
        
        # Update token count
        self.buffer_tokens += tokens
        
        # Maintain buffer based on strategy
        if self.buffer_strategy == "sliding":
            await self._maintain_sliding_buffer()
        elif self.buffer_strategy == "fixed":
            await self._maintain_fixed_buffer()
        else:  # dynamic
            await self._maintain_dynamic_buffer()

    async def _maintain_sliding_buffer(self) -> None:
        """Maintain sliding window buffer."""
        if not self.max_tokens:
            return
            
        # Remove oldest messages until under token limit
        while self.buffer_tokens > self.max_tokens and self.buffer:
            removed = self.buffer.pop(0)
            self.buffer_tokens -= removed["tokens"]

    async def _maintain_fixed_buffer(self) -> None:
        """Maintain fixed-size buffer."""
        if not self.max_messages:
            return
            
        # Remove oldest messages if over limit
        while len(self.buffer) > self.max_messages:
            removed = self.buffer.pop(0)
            self.buffer_tokens -= removed["tokens"]

    async def _maintain_dynamic_buffer(self) -> None:
        """Maintain dynamic buffer based on relevance."""
        if not self.max_tokens:
            return
            
        # Calculate relevance scores
        for item in self.buffer:
            item["relevance"] = self._calculate_relevance(item)
            
        # Sort by relevance
        self.buffer.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Keep most relevant items under token limit
        new_buffer = []
        new_tokens = 0
        
        for item in self.buffer:
            if new_tokens + item["tokens"] <= self.max_tokens:
                new_buffer.append(item)
                new_tokens += item["tokens"]
            else:
                break
                
        # Update buffer
        self.buffer = new_buffer
        self.buffer_tokens = new_tokens

    async def _update_buffer(self) -> None:
        """Update dynamic buffer based on current context."""
        if self.buffer_strategy != "dynamic":
            return
            
        # Get latest summary
        latest_summary = await self.get_latest_summary()
        if not latest_summary:
            return
            
        # Update relevance scores based on summary
        for item in self.buffer:
            item["relevance"] = self._calculate_relevance_to_summary(
                item,
                latest_summary
            )
            
        # Resort buffer
        await self._maintain_dynamic_buffer()
        self.last_buffer_update = datetime.now()

    def _calculate_relevance(self, item: Dict[str, Any]) -> float:
        """Calculate relevance score for an item."""
        # This is a simplified implementation
        # In practice, you would use more sophisticated relevance calculation
        age = (datetime.now() - item["timestamp"]).total_seconds()
        return 1.0 / (1.0 + age / 3600)  # Decay over hours

    def _calculate_relevance_to_summary(
        self,
        item: Dict[str, Any],
        summary: Dict[str, Any]
    ) -> float:
        """Calculate relevance score relative to summary (supports advanced similarity and adaptive threshold)."""
        content = item["message"].get("content", "").lower()
        summary_content = summary["content"].lower()
        # Use custom similarity if set
        if hasattr(self, 'similarity_func') and self.similarity_func:
            sim = self.similarity_func(content, summary_content)
            if hasattr(self, 'adaptive_threshold') and self.adaptive_threshold:
                self.adaptive_threshold.update(sim)
                if sim < self.adaptive_threshold.value:
                    return 0.0
            return sim
        # Default: simple word overlap
        content_words = set(content.split())
        summary_words = set(summary_content.split())
        if not content_words or not summary_words:
            return 0.0
        overlap = len(content_words.intersection(summary_words))
        total = len(content_words.union(summary_words))
        return overlap / total if total > 0 else 0.0

    async def get_buffer_messages(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get messages from buffer."""
        messages = self.buffer[offset:]
        if limit:
            messages = messages[:limit]
        return [m["message"] for m in messages]

    async def get_buffer_with_metadata(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get buffer items with metadata."""
        items = self.buffer[offset:]
        if limit:
            items = items[:limit]
        return items

    async def clear(self) -> None:
        """Clear memory and buffer."""
        await super().clear()
        self.buffer = []
        self.buffer_tokens = 0
        self.last_buffer_update = datetime.now()

    async def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "message_count": len(self.buffer),
            "token_count": self.buffer_tokens,
            "max_tokens": self.max_tokens,
            "buffer_strategy": self.buffer_strategy,
            "last_update": self.last_buffer_update,
            "average_relevance": sum(
                item.get("relevance", 0.0)
                for item in self.buffer
            ) / len(self.buffer) if self.buffer else 0.0
        }

    def set_similarity_func(self, func):
        """Set a custom similarity function for relevance (signature: (a, b) -> float)."""
        self.similarity_func = func

    def set_adaptive_threshold(self, threshold):
        """Set an adaptive threshold instance for filtering/relevance."""
        self.adaptive_threshold = threshold 