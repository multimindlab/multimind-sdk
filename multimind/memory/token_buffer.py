"""
Token-aware memory buffer that can be used with any LLM application, including RAG.
This implementation is similar to LangChain's token buffer but with additional features.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import tiktoken
from .base import BaseMemory

class TokenBufferMemory(BaseMemory):
    """Memory that manages content based on token counts."""

    def __init__(
        self,
        max_tokens: int = 2000,
        token_model: str = "gpt-3.5-turbo",
        prune_strategy: str = "oldest",  # oldest, least_relevant, hybrid
        relevance_threshold: float = 0.7,
        **kwargs
    ):
        """Initialize token buffer memory."""
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.token_model = token_model
        self.prune_strategy = prune_strategy
        self.relevance_threshold = relevance_threshold
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.encoding_for_model(token_model)
        
        # Memory storage
        self.messages: List[Dict[str, Any]] = []
        self.total_tokens = 0
        self.relevance_scores: Dict[str, float] = {}

    async def add_message(
        self,
        message: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a message to memory, pruning if necessary."""
        # Calculate tokens
        content = message.get("content", "")
        tokens = len(self.tokenizer.encode(content))
        
        # Add message
        self.messages.append({
            "message": message,
            "metadata": metadata or {},
            "tokens": tokens,
            "timestamp": datetime.now()
        })
        
        # Update total tokens
        self.total_tokens += tokens
        
        # Prune if needed
        if self.total_tokens > self.max_tokens:
            await self._prune_memory()

    async def get_messages(
        self,
        query: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Get messages, optionally filtered by query and token limit."""
        if not query:
            # Return all messages if no query
            messages = [m["message"] for m in self.messages]
            if max_tokens:
                return self._limit_tokens(messages, max_tokens)
            return messages
            
        # Filter by relevance if query provided
        relevant_messages = []
        current_tokens = 0
        max_tokens = max_tokens or self.max_tokens
        
        for msg in self.messages:
            if current_tokens >= max_tokens:
                break
                
            # Calculate relevance (simplified)
            relevance = self._calculate_relevance(query, msg["message"]["content"])
            if relevance >= self.relevance_threshold:
                relevant_messages.append(msg["message"])
                current_tokens += msg["tokens"]
        
        return relevant_messages

    async def _prune_memory(self) -> None:
        """Prune memory based on strategy."""
        if self.prune_strategy == "oldest":
            await self._prune_oldest()
        elif self.prune_strategy == "least_relevant":
            await self._prune_least_relevant()
        else:  # hybrid
            await self._prune_hybrid()

    async def _prune_oldest(self) -> None:
        """Prune oldest messages first."""
        while self.total_tokens > self.max_tokens and self.messages:
            oldest = self.messages.pop(0)
            self.total_tokens -= oldest["tokens"]

    async def _prune_least_relevant(self) -> None:
        """Prune least relevant messages first."""
        # Sort by relevance
        self.messages.sort(key=lambda x: self.relevance_scores.get(x["message"]["id"], 0))
        
        while self.total_tokens > self.max_tokens and self.messages:
            least_relevant = self.messages.pop(0)
            self.total_tokens -= least_relevant["tokens"]

    async def _prune_hybrid(self) -> None:
        """Hybrid pruning based on both age and relevance."""
        # Calculate combined scores
        for msg in self.messages:
            age = (datetime.now() - msg["timestamp"]).total_seconds()
            relevance = self.relevance_scores.get(msg["message"]["id"], 0.5)
            msg["score"] = (0.7 * relevance) - (0.3 * (age / 3600))  # age in hours
            
        # Sort by combined score
        self.messages.sort(key=lambda x: x["score"])
        
        while self.total_tokens > self.max_tokens and self.messages:
            lowest_score = self.messages.pop(0)
            self.total_tokens -= lowest_score["tokens"]

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content."""
        # This is a simplified implementation
        # In practice, you would use embeddings or other similarity metrics
        query_tokens = set(self.tokenizer.encode(query.lower()))
        content_tokens = set(self.tokenizer.encode(content.lower()))
        
        if not query_tokens or not content_tokens:
            return 0.0
            
        intersection = len(query_tokens.intersection(content_tokens))
        union = len(query_tokens.union(content_tokens))
        
        return intersection / union if union > 0 else 0.0

    def _limit_tokens(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int
    ) -> List[Dict[str, str]]:
        """Limit messages to token count."""
        result = []
        current_tokens = 0
        
        for msg in messages:
            content = msg.get("content", "")
            tokens = len(self.tokenizer.encode(content))
            
            if current_tokens + tokens > max_tokens:
                break
                
            result.append(msg)
            current_tokens += tokens
            
        return result

    async def clear(self) -> None:
        """Clear all messages."""
        self.messages = []
        self.total_tokens = 0
        self.relevance_scores = {} 