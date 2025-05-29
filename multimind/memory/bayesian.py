"""
Nonparametric Bayesian Memory implementation using Dirichlet Process Gaussian Mixture.
"""

from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
from sklearn.mixture import BayesianGaussianMixture
from .base import BaseMemory
from .vector_store import VectorStoreMemory

class BayesianMemory(BaseMemory):
    """Memory implementation using nonparametric Bayesian clustering."""

    def __init__(
        self,
        max_components: int = 10,
        weight_concentration_prior: float = 1.0,
        mean_precision_prior: float = 1.0,
        covariance_prior: float = 1.0,
        **kwargs
    ):
        """Initialize Bayesian memory."""
        super().__init__(**kwargs)
        
        # Clustering parameters
        self.max_components = max_components
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.covariance_prior = covariance_prior
        
        # Component memories
        self.vector_memory = VectorStoreMemory()
        
        # Memory tracking
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.cluster_assignments: Dict[str, int] = {}
        self.cluster_stats: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                'count': 0,
                'mean_embedding': None,
                'covariance': None,
                'weight': 0.0
            }
        )
        
        # Clustering model
        self.mixture_model = BayesianGaussianMixture(
            n_components=max_components,
            weight_concentration_prior=weight_concentration_prior,
            mean_precision_prior=mean_precision_prior,
            covariance_prior=covariance_prior,
            covariance_type='full',
            random_state=42
        )
        
        # Statistics
        self.total_memories = 0
        self.active_clusters = 0
        self.last_cluster_update = datetime.now()

    async def add_memory(
        self,
        memory_id: str,
        content: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new memory with Bayesian clustering."""
        # Create memory entry
        memory = {
            'id': memory_id,
            'content': content,
            'created_at': datetime.now(),
            'last_accessed': datetime.now(),
            'access_count': 0,
            'metadata': metadata or {}
        }
        
        # Store memory
        self.memories[memory_id] = memory
        
        # Get or create embedding
        if embedding is None:
            # This would typically use an embedding model
            embedding = np.random.randn(128)  # Placeholder
        self.embeddings[memory_id] = embedding
        
        # Add to vector memory
        await self.vector_memory.add(memory_id, content, metadata)
        
        # Update clustering
        await self._update_clustering()
        
        self.total_memories += 1

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory by ID."""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            
            # Update access tracking
            memory['access_count'] += 1
            memory['last_accessed'] = datetime.now()
            
            return memory
        return None

    async def get_memories_by_cluster(
        self,
        cluster_id: int,
        include_stats: bool = False
    ) -> List[Dict[str, Any]]:
        """Get memories in a specific cluster."""
        memories = []
        for memory_id, cluster in self.cluster_assignments.items():
            if cluster == cluster_id:
                memory = self.memories[memory_id].copy()
                if include_stats:
                    memory['cluster_stats'] = self.cluster_stats[cluster_id]
                memories.append(memory)
        return memories

    async def get_similar_memories(
        self,
        embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find memories similar to the given embedding."""
        similarities = []
        for memory_id, mem_embedding in self.embeddings.items():
            similarity = np.dot(embedding, mem_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(mem_embedding)
            )
            similarities.append((memory_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k memories
        similar_memories = []
        for memory_id, similarity in similarities[:top_k]:
            memory = self.memories[memory_id].copy()
            memory['similarity'] = similarity
            similar_memories.append(memory)
        
        return similar_memories

    async def get_cluster_stats(
        self,
        cluster_id: int
    ) -> Dict[str, Any]:
        """Get statistics for a cluster."""
        return self.cluster_stats[cluster_id]

    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update an existing memory."""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.update(updates)
            
            # Update vector memory
            if 'content' in updates:
                await self.vector_memory.add(memory_id, updates['content'], memory['metadata'])
            
            # Update embedding if provided
            if 'embedding' in updates:
                self.embeddings[memory_id] = updates['embedding']
                await self._update_clustering()

    async def remove_memory(self, memory_id: str) -> None:
        """Remove a memory."""
        if memory_id in self.memories:
            # Remove from vector memory
            await self.vector_memory.remove(memory_id)
            
            # Remove from tracking
            del self.memories[memory_id]
            if memory_id in self.embeddings:
                del self.embeddings[memory_id]
            if memory_id in self.cluster_assignments:
                del self.cluster_assignments[memory_id]
            
            # Update clustering
            await self._update_clustering()
            
            self.total_memories -= 1

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_memories': self.total_memories,
            'active_clusters': self.active_clusters,
            'avg_cluster_size': self.total_memories / self.active_clusters if self.active_clusters > 0 else 0,
            'mixture_weights': self.mixture_model.weights_ if hasattr(self.mixture_model, 'weights_') else []
        }

    async def _update_clustering(self) -> None:
        """Update the clustering model."""
        if not self.embeddings:
            return
            
        # Prepare data
        embeddings = np.array(list(self.embeddings.values()))
        
        # Fit mixture model
        self.mixture_model.fit(embeddings)
        
        # Update cluster assignments
        cluster_labels = self.mixture_model.predict(embeddings)
        for memory_id, label in zip(self.embeddings.keys(), cluster_labels):
            self.cluster_assignments[memory_id] = label
        
        # Update cluster statistics
        self.cluster_stats.clear()
        for i in range(self.mixture_model.n_components_):
            if i in cluster_labels:
                mask = cluster_labels == i
                cluster_embeddings = embeddings[mask]
                
                self.cluster_stats[i] = {
                    'count': np.sum(mask),
                    'mean_embedding': self.mixture_model.means_[i],
                    'covariance': self.mixture_model.covariances_[i],
                    'weight': self.mixture_model.weights_[i]
                }
        
        self.active_clusters = len(self.cluster_stats)
        self.last_cluster_update = datetime.now() 