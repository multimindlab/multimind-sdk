"""
Vector store memory implementation for semantic search.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class VectorStoreMemory(BaseMemory):
    """Memory that uses vector stores for semantic search."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        vector_dim: int = 1536,  # Default for OpenAI embeddings
        storage_path: Optional[str] = None,
        similarity_threshold: float = 0.7,
        batch_size: int = 10  # For batch processing
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.vector_dim = vector_dim
        self.storage_path = Path(storage_path) if storage_path else None
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.messages: List[Dict[str, str]] = []
        self.vectors: List[List[float]] = []
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message and its embedding to the vector store."""
        message_with_timestamp = {
            **message,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message_with_timestamp)
        
        # Get embedding for the message
        embedding = await self._get_embedding(message["content"])
        self.vectors.append(embedding)
        
        await self.save()

    async def add_messages(self, messages: List[Dict[str, str]]) -> None:
        """Add multiple messages in batch."""
        # Prepare messages with timestamps
        messages_with_timestamps = [
            {**msg, "timestamp": datetime.now().isoformat()}
            for msg in messages
        ]
        
        # Get embeddings in batches
        for i in range(0, len(messages), self.batch_size):
            batch = messages[i:i + self.batch_size]
            embeddings = await self._get_embeddings_batch([m["content"] for m in batch])
            
            # Add to storage
            self.messages.extend(messages_with_timestamps[i:i + self.batch_size])
            self.vectors.extend(embeddings)
        
        await self.save()

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages."""
        return self.messages

    async def clear(self) -> None:
        """Clear all messages and vectors."""
        self.messages.clear()
        self.vectors.clear()
        await self.save()

    async def save(self) -> None:
        """Save messages and vectors to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "messages": self.messages,
                    "vectors": self.vectors
                }, f)

    def load(self) -> None:
        """Load messages and vectors from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.messages = data.get("messages", [])
                self.vectors = data.get("vectors", [])

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using the LLM."""
        try:
            embedding = await self.llm.embeddings(text)
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0.0] * self.vector_dim

    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in batch."""
        try:
            embeddings = await self.llm.embeddings_batch(texts)
            return embeddings
        except Exception as e:
            print(f"Error getting batch embeddings: {e}")
            return [[0.0] * self.vector_dim for _ in texts]

    async def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar messages using vector similarity."""
        if not self.messages or not self.vectors:
            return []

        # Get query embedding
        query_embedding = await self._get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for vec in self.vectors:
            similarity = self._cosine_similarity(query_embedding, vec)
            similarities.append(similarity)
        
        # Get top k similar messages
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            if similarities[idx] >= self.similarity_threshold:
                results.append({
                    "message": self.messages[idx],
                    "similarity": similarities[idx]
                })
        
        return results

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    async def get_relevant_context(self, query: str, k: int = 5) -> str:
        """Get relevant context for a query."""
        similar_messages = await self.search_similar(query, k)
        if not similar_messages:
            return ""
        
        # Format context from similar messages
        context = []
        for result in similar_messages:
            msg = result["message"]
            context.append(f"{msg['role']}: {msg['content']}")
        
        return "\n".join(context)

    async def get_similarity_matrix(self) -> np.ndarray:
        """Get similarity matrix between all stored messages."""
        if not self.vectors:
            return np.array([])
        
        vectors = np.array(self.vectors)
        # Normalize vectors
        normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        # Calculate similarity matrix
        return np.dot(normalized, normalized.T)

    async def get_clusters(self, threshold: float = 0.8) -> List[List[int]]:
        """Get clusters of similar messages using the similarity matrix."""
        if not self.vectors:
            return []
        
        similarity_matrix = await self.get_similarity_matrix()
        clusters = []
        used_indices = set()
        
        for i in range(len(similarity_matrix)):
            if i in used_indices:
                continue
            
            # Find similar messages
            similar_indices = np.where(similarity_matrix[i] >= threshold)[0]
            if len(similar_indices) > 1:  # Only consider clusters with multiple messages
                clusters.append(similar_indices.tolist())
                used_indices.update(similar_indices)
        
        return clusters

    async def get_cluster_summaries(self, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Get summaries of message clusters."""
        clusters = await self.get_clusters(threshold)
        summaries = []
        
        for cluster in clusters:
            # Get messages in cluster
            cluster_messages = [self.messages[i] for i in cluster]
            
            # Create summary
            summary = {
                "size": len(cluster),
                "messages": cluster_messages,
                "average_similarity": np.mean([
                    self._cosine_similarity(self.vectors[i], self.vectors[j])
                    for i in cluster
                    for j in cluster
                    if i != j
                ]) if len(cluster) > 1 else 1.0
            }
            summaries.append(summary)
        
        return summaries 