"""
Advanced memory system with episodic and semantic memory support.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import asyncio
import numpy as np
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
from ..models.base import BaseLLM

@dataclass
class MemoryItem:
    """Base class for memory items."""
    content: str
    timestamp: float
    importance: float
    tokens: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class EpisodicMemory(MemoryItem):
    """Represents an episodic memory item."""
    event_type: str
    context: Dict[str, Any]
    emotions: List[str]
    participants: List[str]
    location: Optional[str]
    duration: Optional[float]

@dataclass
class SemanticMemory(MemoryItem):
    """Represents a semantic memory item."""
    concept: str
    relationships: List[Dict[str, Any]]
    attributes: Dict[str, Any]
    category: str
    confidence: float

@dataclass
class WorkingMemory(MemoryItem):
    """Represents a working memory item."""
    priority: float
    expiration: Optional[float]
    dependencies: List[str]
    state: str

class MemoryType(Enum):
    """Types of memory."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"

class MemoryCompressionStrategy(Enum):
    """Strategies for memory compression."""
    IMPORTANCE = "importance"
    RECENCY = "recency"
    RELEVANCE = "relevance"
    HYBRID = "hybrid"

class AdvancedMemory:
    """Advanced memory system with multiple memory types and compression."""

    def __init__(
        self,
        model: BaseLLM,
        max_tokens: int = 4000,
        compression_threshold: float = 0.8,
        **kwargs
    ):
        """
        Initialize advanced memory system.
        
        Args:
            model: Language model
            max_tokens: Maximum tokens for memory
            compression_threshold: Threshold for memory compression
            **kwargs: Additional parameters
        """
        self.model = model
        self.max_tokens = max_tokens
        self.compression_threshold = compression_threshold
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.embedding_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        
        # Initialize memory stores
        self.episodic_memory: List[EpisodicMemory] = []
        self.semantic_memory: List[SemanticMemory] = []
        self.working_memory: List[WorkingMemory] = []
        
        # Initialize compression state
        self.compression_state = {
            "last_compression": datetime.now(),
            "compression_count": 0,
            "total_tokens_compressed": 0
        }
        
        self.kwargs = kwargs

    async def add_to_memory(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Add content to memory.
        
        Args:
            content: Content to remember
            memory_type: Type of memory to use
            metadata: Optional metadata
            **kwargs: Additional parameters
        """
        # Calculate tokens and importance
        tokens = len(self.tokenizer.encode(content))
        importance = await self._calculate_importance(content, **kwargs)
        
        # Generate embedding
        embedding = await self._generate_embedding(content)
        
        # Create memory item based on type
        if memory_type == MemoryType.EPISODIC:
            memory_item = await self._create_episodic_memory(
                content=content,
                tokens=tokens,
                importance=importance,
                metadata=metadata,
                embedding=embedding,
                **kwargs
            )
            self.episodic_memory.append(memory_item)
        
        elif memory_type == MemoryType.SEMANTIC:
            memory_item = await self._create_semantic_memory(
                content=content,
                tokens=tokens,
                importance=importance,
                metadata=metadata,
                embedding=embedding,
                **kwargs
            )
            self.semantic_memory.append(memory_item)
        
        else:
            memory_item = await self._create_working_memory(
                content=content,
                tokens=tokens,
                importance=importance,
                metadata=metadata,
                embedding=embedding,
                **kwargs
            )
            self.working_memory.append(memory_item)
        
        # Check if compression is needed
        await self._check_compression()

    async def get_relevant_memory(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        k: int = 5,
        **kwargs
    ) -> List[MemoryItem]:
        """
        Retrieve relevant memory items.
        
        Args:
            query: Query to find relevant memories
            memory_types: Optional list of memory types to search
            k: Number of items to retrieve
            **kwargs: Additional parameters
            
        Returns:
            List of relevant memory items
        """
        if memory_types is None:
            memory_types = list(MemoryType)
        
        # Get items from specified memory types
        all_items = []
        for memory_type in memory_types:
            if memory_type == MemoryType.EPISODIC:
                all_items.extend(self.episodic_memory)
            elif memory_type == MemoryType.SEMANTIC:
                all_items.extend(self.semantic_memory)
            else:
                all_items.extend(self.working_memory)
        
        if not all_items:
            return []
        
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        # Calculate relevance scores
        scores = []
        for item in all_items:
            # Calculate semantic similarity
            semantic_score = self._cosine_similarity(
                query_embedding,
                item.embedding
            )
            
            # Calculate importance score
            importance_score = item.importance
            
            # Calculate recency score
            recency_score = self._calculate_recency_score(item)
            
            # Combine scores
            combined_score = (
                0.4 * semantic_score +
                0.3 * importance_score +
                0.3 * recency_score
            )
            
            scores.append(combined_score)
        
        # Get top k items
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return [all_items[i] for i in top_k_indices]

    async def compress_memory(
        self,
        strategy: MemoryCompressionStrategy = MemoryCompressionStrategy.HYBRID,
        **kwargs
    ) -> None:
        """
        Compress memory using specified strategy.
        
        Args:
            strategy: Compression strategy to use
            **kwargs: Additional parameters
        """
        # Get all memory items
        all_items = (
            self.episodic_memory +
            self.semantic_memory +
            self.working_memory
        )
        
        if not all_items:
            return
        
        # Calculate compression scores
        compression_scores = []
        for item in all_items:
            if strategy == MemoryCompressionStrategy.IMPORTANCE:
                score = 1 - item.importance
            elif strategy == MemoryCompressionStrategy.RECENCY:
                score = self._calculate_recency_score(item)
            elif strategy == MemoryCompressionStrategy.RELEVANCE:
                score = await self._calculate_relevance_score(item, **kwargs)
            else:  # HYBRID
                importance_score = 1 - item.importance
                recency_score = self._calculate_recency_score(item)
                relevance_score = await self._calculate_relevance_score(
                    item,
                    **kwargs
                )
                score = (
                    0.4 * importance_score +
                    0.3 * recency_score +
                    0.3 * relevance_score
                )
            
            compression_scores.append(score)
        
        # Sort items by compression score
        sorted_items = [
            item for _, item in sorted(
                zip(compression_scores, all_items)
            )
        ]
        
        # Compress items until under token budget
        total_tokens = sum(item.tokens for item in all_items)
        compressed_items = []
        
        for item in sorted_items:
            if total_tokens <= self.max_tokens:
                break
            
            # Compress item
            compressed_item = await self._compress_item(item, **kwargs)
            compressed_items.append(compressed_item)
            
            # Update total tokens
            total_tokens -= (item.tokens - compressed_item.tokens)
        
        # Update memory stores
        self._update_memory_stores(compressed_items)
        
        # Update compression state
        self.compression_state["last_compression"] = datetime.now()
        self.compression_state["compression_count"] += 1
        self.compression_state["total_tokens_compressed"] += (
            sum(item.tokens for item in all_items) -
            sum(item.tokens for item in compressed_items)
        )

    async def _create_episodic_memory(
        self,
        content: str,
        tokens: int,
        importance: float,
        metadata: Optional[Dict[str, Any]],
        embedding: List[float],
        **kwargs
    ) -> EpisodicMemory:
        """Create episodic memory item."""
        # Extract event information
        event_info = await self._extract_event_info(content, **kwargs)
        
        return EpisodicMemory(
            content=content,
            timestamp=datetime.now().timestamp(),
            importance=importance,
            tokens=tokens,
            metadata=metadata or {},
            embedding=embedding,
            event_type=event_info["type"],
            context=event_info["context"],
            emotions=event_info["emotions"],
            participants=event_info["participants"],
            location=event_info.get("location"),
            duration=event_info.get("duration")
        )

    async def _create_semantic_memory(
        self,
        content: str,
        tokens: int,
        importance: float,
        metadata: Optional[Dict[str, Any]],
        embedding: List[float],
        **kwargs
    ) -> SemanticMemory:
        """Create semantic memory item."""
        # Extract semantic information
        semantic_info = await self._extract_semantic_info(content, **kwargs)
        
        return SemanticMemory(
            content=content,
            timestamp=datetime.now().timestamp(),
            importance=importance,
            tokens=tokens,
            metadata=metadata or {},
            embedding=embedding,
            concept=semantic_info["concept"],
            relationships=semantic_info["relationships"],
            attributes=semantic_info["attributes"],
            category=semantic_info["category"],
            confidence=semantic_info["confidence"]
        )

    async def _create_working_memory(
        self,
        content: str,
        tokens: int,
        importance: float,
        metadata: Optional[Dict[str, Any]],
        embedding: List[float],
        **kwargs
    ) -> WorkingMemory:
        """Create working memory item."""
        return WorkingMemory(
            content=content,
            timestamp=datetime.now().timestamp(),
            importance=importance,
            tokens=tokens,
            metadata=metadata or {},
            embedding=embedding,
            priority=kwargs.get("priority", 0.5),
            expiration=kwargs.get("expiration"),
            dependencies=kwargs.get("dependencies", []),
            state=kwargs.get("state", "active")
        )

    async def _extract_event_info(
        self,
        content: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract event information from content."""
        # Use LLM to extract event information
        prompt = f"""
        Extract event information from the following content.
        Provide:
        1. Event type
        2. Context
        3. Emotions
        4. Participants
        5. Location (if any)
        6. Duration (if any)
        
        Content:
        {content}
        """
        
        response = await self.model.generate(prompt=prompt, **kwargs)
        # Parse response into event info
        # This is a placeholder implementation
        return {
            "type": "general",
            "context": {},
            "emotions": [],
            "participants": []
        }

    async def _extract_semantic_info(
        self,
        content: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract semantic information from content."""
        # Use LLM to extract semantic information
        prompt = f"""
        Extract semantic information from the following content.
        Provide:
        1. Main concept
        2. Relationships
        3. Attributes
        4. Category
        5. Confidence
        
        Content:
        {content}
        """
        
        response = await self.model.generate(prompt=prompt, **kwargs)
        # Parse response into semantic info
        # This is a placeholder implementation
        return {
            "concept": "general",
            "relationships": [],
            "attributes": {},
            "category": "general",
            "confidence": 0.8
        }

    async def _generate_embedding(
        self,
        text: str
    ) -> List[float]:
        """Generate embedding for text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        
        return outputs.last_hidden_state.mean(dim=1).numpy()[0].tolist()

    async def _calculate_importance(
        self,
        content: str,
        **kwargs
    ) -> float:
        """Calculate importance score for content."""
        # Use LLM to calculate importance
        prompt = f"""
        Rate the importance of the following content on a scale of 0 to 1.
        Consider:
        1. Information value
        2. Uniqueness
        3. Relevance
        4. Impact
        
        Content:
        {content}
        """
        
        response = await self.model.generate(prompt=prompt, **kwargs)
        # Parse response into importance score
        # This is a placeholder implementation
        return 0.5

    async def _calculate_relevance_score(
        self,
        item: MemoryItem,
        **kwargs
    ) -> float:
        """Calculate relevance score for memory item."""
        # This is a placeholder implementation
        return 0.5

    def _calculate_recency_score(
        self,
        item: MemoryItem
    ) -> float:
        """Calculate recency score for memory item."""
        current_time = datetime.now().timestamp()
        time_diff = current_time - item.timestamp
        return np.exp(-time_diff / (24 * 3600))  # Decay over days

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )

    async def _compress_item(
        self,
        item: MemoryItem,
        **kwargs
    ) -> MemoryItem:
        """Compress memory item."""
        # Use LLM to compress content
        prompt = f"""
        Compress the following content while preserving key information.
        Make it more concise but maintain important details.
        
        Content:
        {item.content}
        """
        
        response = await self.model.generate(prompt=prompt, **kwargs)
        
        # Create new memory item with compressed content
        compressed_tokens = len(self.tokenizer.encode(response))
        compressed_embedding = await self._generate_embedding(response)
        
        if isinstance(item, EpisodicMemory):
            return EpisodicMemory(
                content=response,
                timestamp=item.timestamp,
                importance=item.importance,
                tokens=compressed_tokens,
                metadata=item.metadata,
                embedding=compressed_embedding,
                event_type=item.event_type,
                context=item.context,
                emotions=item.emotions,
                participants=item.participants,
                location=item.location,
                duration=item.duration
            )
        elif isinstance(item, SemanticMemory):
            return SemanticMemory(
                content=response,
                timestamp=item.timestamp,
                importance=item.importance,
                tokens=compressed_tokens,
                metadata=item.metadata,
                embedding=compressed_embedding,
                concept=item.concept,
                relationships=item.relationships,
                attributes=item.attributes,
                category=item.category,
                confidence=item.confidence
            )
        else:
            return WorkingMemory(
                content=response,
                timestamp=item.timestamp,
                importance=item.importance,
                tokens=compressed_tokens,
                metadata=item.metadata,
                embedding=compressed_embedding,
                priority=item.priority,
                expiration=item.expiration,
                dependencies=item.dependencies,
                state=item.state
            )

    def _update_memory_stores(
        self,
        compressed_items: List[MemoryItem]
    ) -> None:
        """Update memory stores with compressed items."""
        # Clear existing stores
        self.episodic_memory.clear()
        self.semantic_memory.clear()
        self.working_memory.clear()
        
        # Add compressed items to appropriate stores
        for item in compressed_items:
            if isinstance(item, EpisodicMemory):
                self.episodic_memory.append(item)
            elif isinstance(item, SemanticMemory):
                self.semantic_memory.append(item)
            else:
                self.working_memory.append(item)

    async def _check_compression(self) -> None:
        """Check if memory compression is needed."""
        total_tokens = (
            sum(item.tokens for item in self.episodic_memory) +
            sum(item.tokens for item in self.semantic_memory) +
            sum(item.tokens for item in self.working_memory)
        )
        
        if total_tokens > self.max_tokens * self.compression_threshold:
            await self.compress_memory() 