"""
Advanced memory system with episodic and semantic memory support.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

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

    # Defaults required: the base class has a defaulted field (embedding)
    event_type: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    emotions: List[str] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    location: Optional[str] = None
    duration: Optional[float] = None


@dataclass
class SemanticMemory(MemoryItem):
    """Represents a semantic memory item."""

    concept: str = ""
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    category: str = ""
    confidence: float = 0.0


@dataclass
class WorkingMemory(MemoryItem):
    """Represents a working memory item."""

    priority: float = 0.0
    expiration: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    state: str = ""


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
        self, model: BaseLLM, max_tokens: int = 4000, compression_threshold: float = 0.8, **kwargs
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
        # Lazy-load heavyweight HF artifacts (tokenizer + embedding model) to avoid
        # downloading models during __init__ / import-time.
        self.tokenizer = None
        self.embedding_model = None
        self._device: Optional[str] = None
        self._models_lock = asyncio.Lock()
        # Cache computed embeddings to avoid recomputation on repeated texts.
        self._embedding_cache: Dict[str, List[float]] = {}
        self._embedding_cache_lock = asyncio.Lock()

        # Initialize memory stores
        self.episodic_memory: List[EpisodicMemory] = []
        self.semantic_memory: List[SemanticMemory] = []
        self.working_memory: List[WorkingMemory] = []

        # Initialize compression state
        self.compression_state = {
            "last_compression": datetime.now(),
            "compression_count": 0,
            "total_tokens_compressed": 0,
        }

        self.kwargs = kwargs

    async def _ensure_embedding_models_loaded(self) -> None:
        """Lazily load tokenizer + embedding model on first real use."""
        if self.tokenizer is not None and self.embedding_model is not None:
            return

        async with self._models_lock:
            if self.tokenizer is not None and self.embedding_model is not None:
                return

            def _load():
                model_name = "sentence-transformers/all-mpnet-base-v2"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                embedding_model = AutoModel.from_pretrained(model_name)
                device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
                embedding_model.to(device)
                embedding_model.eval()
                return tokenizer, embedding_model

            # Decide device once on first use (and reuse thereafter).
            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer, self.embedding_model = await asyncio.to_thread(_load)

    async def add_to_memory(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Add content to memory.

        Args:
            content: Content to remember
            memory_type: Type of memory to use
            metadata: Optional metadata
            **kwargs: Additional parameters
        """
        await self._ensure_embedding_models_loaded()
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
                **kwargs,
            )
            self.episodic_memory.append(memory_item)

        elif memory_type == MemoryType.SEMANTIC:
            memory_item = await self._create_semantic_memory(
                content=content,
                tokens=tokens,
                importance=importance,
                metadata=metadata,
                embedding=embedding,
                **kwargs,
            )
            self.semantic_memory.append(memory_item)

        else:
            memory_item = await self._create_working_memory(
                content=content,
                tokens=tokens,
                importance=importance,
                metadata=metadata,
                embedding=embedding,
                **kwargs,
            )
            self.working_memory.append(memory_item)

        # Check if compression is needed
        await self._check_compression()

    async def get_relevant_memory(
        self, query: str, memory_types: Optional[List[MemoryType]] = None, k: int = 5, **kwargs
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
            semantic_score = self._cosine_similarity(query_embedding, item.embedding)

            # Calculate importance score
            importance_score = item.importance

            # Calculate recency score
            recency_score = self._calculate_recency_score(item)

            # Combine scores
            combined_score = 0.4 * semantic_score + 0.3 * importance_score + 0.3 * recency_score

            scores.append(combined_score)

        # Get top k items
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return [all_items[i] for i in top_k_indices]

    async def compress_memory(
        self, strategy: MemoryCompressionStrategy = MemoryCompressionStrategy.HYBRID, **kwargs
    ) -> None:
        """
        Compress memory using specified strategy.

        Args:
            strategy: Compression strategy to use
            **kwargs: Additional parameters
        """
        # Get all memory items
        all_items = self.episodic_memory + self.semantic_memory + self.working_memory

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
                relevance_score = await self._calculate_relevance_score(item, **kwargs)
                score = 0.4 * importance_score + 0.3 * recency_score + 0.3 * relevance_score

            compression_scores.append(score)

        # Sort items by compression score
        sorted_items = [item for _, item in sorted(zip(compression_scores, all_items))]

        # Compress items until under token budget.
        # Important: keep both compressed items and remaining uncompressed items.
        total_tokens = sum(item.tokens for item in all_items)
        new_items: List[MemoryItem] = []
        compressed_items: List[MemoryItem] = []

        for idx, item in enumerate(sorted_items):
            if total_tokens <= self.max_tokens:
                # Preserve the rest (uncompressed) to avoid losing information.
                new_items.extend(sorted_items[idx:])
                break

            compressed_item = await self._compress_item(item, **kwargs)
            compressed_items.append(compressed_item)
            new_items.append(compressed_item)

            # Update total tokens
            total_tokens -= item.tokens - compressed_item.tokens

        if not new_items:
            new_items = list(all_items)

        # Update memory stores with both compressed + untouched items.
        self._update_memory_stores(new_items)

        # Update compression state
        self.compression_state["last_compression"] = datetime.now()
        self.compression_state["compression_count"] += 1
        self.compression_state["total_tokens_compressed"] += sum(
            item.tokens for item in all_items
        ) - sum(item.tokens for item in new_items)

    async def _create_episodic_memory(
        self,
        content: str,
        tokens: int,
        importance: float,
        metadata: Optional[Dict[str, Any]],
        embedding: List[float],
        **kwargs,
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
            duration=event_info.get("duration"),
        )

    async def _create_semantic_memory(
        self,
        content: str,
        tokens: int,
        importance: float,
        metadata: Optional[Dict[str, Any]],
        embedding: List[float],
        **kwargs,
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
            confidence=semantic_info["confidence"],
        )

    async def _create_working_memory(
        self,
        content: str,
        tokens: int,
        importance: float,
        metadata: Optional[Dict[str, Any]],
        embedding: List[float],
        **kwargs,
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
            state=kwargs.get("state", "active"),
        )

    async def _extract_event_info(self, content: str, **kwargs) -> Dict[str, Any]:
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
        return {"type": "general", "context": {}, "emotions": [], "participants": []}

    async def _extract_semantic_info(self, content: str, **kwargs) -> Dict[str, Any]:
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
            "confidence": 0.8,
        }

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        await self._ensure_embedding_models_loaded()
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached

        async with self._embedding_cache_lock:
            cached = self._embedding_cache.get(text)
            if cached is not None:
                return cached

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
            )

            # Ensure tensors live on the same device as the embedding model.
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.embedding_model(**inputs)

            token_embeddings = outputs.last_hidden_state  # [batch, seq, hidden]
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones(
                    token_embeddings.shape[:2],
                    device=token_embeddings.device,
                    dtype=token_embeddings.dtype,
                )

            # Sentence-Transformers style mean pooling with attention mask.
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
            sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask

            emb = pooled[0].detach().cpu().numpy().astype(float).tolist()
            self._embedding_cache[text] = emb
            return emb

    def close(self) -> None:
        """Best-effort cleanup for long-running services."""
        self._embedding_cache.clear()
        self.tokenizer = None
        self.embedding_model = None
        self._device = None

    async def _calculate_importance(self, content: str, **kwargs) -> float:
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

    async def _calculate_relevance_score(self, item: MemoryItem, **kwargs) -> float:
        """Calculate relevance score for memory item."""
        # This is a placeholder implementation
        return 0.5

    def _calculate_recency_score(self, item: MemoryItem) -> float:
        """Calculate recency score for memory item."""
        current_time = datetime.now().timestamp()
        time_diff = current_time - item.timestamp
        return np.exp(-time_diff / (24 * 3600))  # Decay over days

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    async def _compress_item(self, item: MemoryItem, **kwargs) -> MemoryItem:
        """Compress memory item."""
        await self._ensure_embedding_models_loaded()
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
                duration=item.duration,
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
                confidence=item.confidence,
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
                state=item.state,
            )

    def _update_memory_stores(self, compressed_items: List[MemoryItem]) -> None:
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
            sum(item.tokens for item in self.episodic_memory)
            + sum(item.tokens for item in self.semantic_memory)
            + sum(item.tokens for item in self.working_memory)
        )

        if total_tokens > self.max_tokens * self.compression_threshold:
            await self.compress_memory()
