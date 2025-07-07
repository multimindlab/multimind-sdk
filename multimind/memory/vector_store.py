"""
Vector store memory implementation that uses the vector store interface.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory
from ..vector_store.vector_store import VectorStore
from ..vector_store.base import VectorStoreConfig, VectorStoreType
import os

class VectorStoreMemory(BaseMemory):
    """Memory that uses vector store for storing and retrieving embeddings."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        vector_store_config: Optional[VectorStoreConfig] = None,
        similarity_threshold: float = 0.7,
        enable_metadata: bool = True,
        enable_backup: bool = True,
        backup_interval: int = 3600,  # 1 hour
        max_backups: int = 5,
        enable_pruning: bool = True,
        pruning_threshold: float = 0.5,
        pruning_interval: int = 3600  # 1 hour
    ):
        """Initialize vector store memory."""
        super().__init__(memory_key)
        self.llm = llm
        self.similarity_threshold = similarity_threshold
        self.enable_metadata = enable_metadata
        self.enable_backup = enable_backup
        self.backup_interval = backup_interval
        self.max_backups = max_backups
        self.enable_pruning = enable_pruning
        self.pruning_threshold = pruning_threshold
        self.pruning_interval = pruning_interval

        # Initialize vector store with default config if none provided
        if vector_store_config is None:
            vector_store_config = VectorStoreConfig(
                store_type=VectorStoreType.FAISS,  # Default to FAISS
                vector_dim=1536,  # Default for OpenAI embeddings
                max_vectors=10000,
                storage_path=None,  # Will be set if needed
                index_type="flat",  # Default index type
                enable_compression=True,
                compression_threshold=0.8,
                enable_quantization=False,
                quantization_bits=8
            )
        
        self.vector_store = VectorStore(vector_store_config)
        self.last_backup = datetime.now()
        self.backup_history: List[Dict[str, Any]] = []
        self.last_pruning = datetime.now()

        # Initialize vector store
        if vector_store_config.storage_path:
            self.vector_store.load(vector_store_config.storage_path)

    async def add(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a vector to the store."""
        # Generate embedding
        embedding = await self._get_embedding(content)
        
        # Prepare metadata
        memory_metadata = {
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        } if self.enable_metadata else {}

        # Add to vector store using the correct method
        await self.vector_store.add_vectors(
            vectors=[embedding.tolist()],
            metadatas=[memory_metadata],
            documents=[{"content": content}],
            ids=[memory_id]
        )

        # Check if pruning needed
        if (
            self.enable_pruning and
            (datetime.now() - self.last_pruning).total_seconds() >= self.pruning_interval
        ):
            await self._prune_vectors()

        # Check if backup needed
        if (
            self.enable_backup and
            (datetime.now() - self.last_backup).total_seconds() >= self.backup_interval
        ):
            await self._backup()

    async def get(
        self,
        memory_id: str,
        update_access: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get a vector by ID."""
        # Search for exact ID using search with filter
        results = await self.vector_store.search(
            query_vector=[0] * self.vector_store.config.vector_dim,  # Dummy vector
            k=1,
            filter_criteria={"id": memory_id}
        )
        return results[0] if results else None

    async def search(
        self,
        query: str,
        k: int = 5,
        filter_func: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        # Generate query embedding
        query_embedding = await self._get_embedding(query)
        
        # Search using the correct method
        results = await self.vector_store.search(
            query_vector=query_embedding.tolist(),
            k=k,
            filter_criteria=filter_func.__dict__ if filter_func else None
        )
        return results

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using LLM."""
        # Use the actual LLM to generate embeddings
        return await self.llm.get_embedding(text)

    async def _prune_vectors(self) -> None:
        """Prune vectors based on access patterns."""
        if not self.enable_pruning:
            return
        
        # Get all vectors and find ones to prune
        results = await self.vector_store.search(
            query_vector=[0] * self.vector_store.config.vector_dim,
            k=self.vector_store.config.max_vectors
        )
        
        to_prune = []
        for result in results:
            metadata = result.metadata
            if not metadata:
                continue
            
            last_access = datetime.fromisoformat(metadata.get("last_access", "2000-01-01"))
            access_count = metadata.get("access_count", 0)
            
            if (datetime.now() - last_access).total_seconds() > self.pruning_interval or \
               access_count < self.pruning_threshold:
                to_prune.append(result.id)
        
        if to_prune:
            await self.vector_store.delete_vectors(to_prune)
        
        self.last_pruning = datetime.now()

    async def _backup(self) -> None:
        """Create a backup of the current state."""
        if not self.enable_backup or not self.vector_store.config.storage_path:
            return
        
        backup_path = f"{self.vector_store.config.storage_path}/backup_{datetime.now().isoformat()}"
        await self.vector_store.persist(backup_path)
        
        self.backup_history.append({
            "path": backup_path,
            "timestamp": datetime.now().isoformat()
        })
        
        # Remove old backups
        if len(self.backup_history) > self.max_backups:
            old_backup = self.backup_history.pop(0)
            # Delete old backup file
            try:
                if os.path.exists(old_backup["path"]):
                    os.remove(old_backup["path"])
            except Exception as e:
                print(f"Warning: Failed to delete old backup file {old_backup['path']}: {e}")
        
        self.last_backup = datetime.now()

    def clear(self) -> None:
        """Clear all vectors."""
        self.vector_store.clear()
        self.last_backup = datetime.now()
        self.backup_history = []
        self.last_pruning = datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return self.vector_store.get_stats() 