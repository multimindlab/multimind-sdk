"""
Differentially-Private Federated Memory implementation.
"""

from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import torch
from torch import nn
from .base import BaseMemory
from .vector_store import VectorStoreMemory

class DPNoiseGenerator:
    """Differential Privacy noise generator."""
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity

    def add_noise(self, data: np.ndarray) -> np.ndarray:
        """Add calibrated noise to data."""
        scale = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, scale, data.shape)
        return data + noise

class FederatedMemory(BaseMemory):
    """Memory implementation with differential privacy and federated learning."""

    def __init__(
        self,
        num_clients: int = 5,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        aggregation_rounds: int = 10,
        local_epochs: int = 3,
        batch_size: int = 32,
        **kwargs
    ):
        """Initialize federated memory."""
        super().__init__(**kwargs)
        
        # Privacy parameters
        self.epsilon = epsilon
        self.delta = delta
        
        # Federated learning parameters
        self.num_clients = num_clients
        self.aggregation_rounds = aggregation_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        
        # Component memories
        self.vector_memory = VectorStoreMemory()
        
        # Client memories
        self.client_memories: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.client_embeddings: Dict[int, Dict[str, np.ndarray]] = defaultdict(dict)
        
        # Global model
        self.global_model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Privacy components
        self.noise_generator = DPNoiseGenerator(
            epsilon=epsilon,
            delta=delta
        )
        
        # Statistics
        self.total_memories = 0
        self.aggregation_history = []
        self.privacy_budget_used = 0.0

    async def add_memory(
        self,
        memory_id: str,
        content: str,
        client_id: int,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new memory to a specific client."""
        # Create memory entry
        memory = {
            'id': memory_id,
            'content': content,
            'client_id': client_id,
            'created_at': datetime.now(),
            'last_accessed': datetime.now(),
            'access_count': 0,
            'metadata': metadata or {}
        }
        
        # Get or create embedding
        if embedding is None:
            # This would typically use an embedding model
            embedding = np.random.randn(128)  # Placeholder
        
        # Add noise to embedding for privacy
        noisy_embedding = self.noise_generator.add_noise(embedding)
        
        # Store in client memory
        self.client_memories[client_id][memory_id] = memory
        self.client_embeddings[client_id][memory_id] = noisy_embedding
        
        # Add to vector memory
        await self.vector_memory.add(memory_id, content, metadata)
        
        self.total_memories += 1
        self.privacy_budget_used += self.epsilon

    async def get_memory(
        self,
        memory_id: str,
        client_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Get a memory by ID, optionally from a specific client."""
        if client_id is not None:
            if client_id in self.client_memories and memory_id in self.client_memories[client_id]:
                memory = self.client_memories[client_id][memory_id]
                memory['access_count'] += 1
                memory['last_accessed'] = datetime.now()
                return memory
        else:
            # Search across all clients
            for client_memories in self.client_memories.values():
                if memory_id in client_memories:
                    memory = client_memories[memory_id]
                    memory['access_count'] += 1
                    memory['last_accessed'] = datetime.now()
                    return memory
        return None

    async def get_client_memories(
        self,
        client_id: int
    ) -> List[Dict[str, Any]]:
        """Get all memories for a specific client."""
        if client_id in self.client_memories:
            return list(self.client_memories[client_id].values())
        return []

    async def train_federated_model(self) -> None:
        """Train the global model using federated learning."""
        for round in range(self.aggregation_rounds):
            # Train local models
            local_updates = []
            for client_id in range(self.num_clients):
                if client_id in self.client_embeddings:
                    # Train local model
                    local_model = self._train_local_model(
                        client_id,
                        self.local_epochs
                    )
                    local_updates.append(local_model)
            
            # Aggregate updates with privacy
            if local_updates:
                self._aggregate_updates(local_updates)
                
                # Record aggregation
                self.aggregation_history.append({
                    'round': round,
                    'num_clients': len(local_updates),
                    'timestamp': datetime.now()
                })

    async def get_similar_memories(
        self,
        embedding: np.ndarray,
        client_id: Optional[int] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find memories similar to the given embedding."""
        similarities = []
        
        if client_id is not None:
            # Search in specific client
            if client_id in self.client_embeddings:
                for memory_id, client_embedding in self.client_embeddings[client_id].items():
                    similarity = np.dot(embedding, client_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(client_embedding)
                    )
                    similarities.append((client_id, memory_id, similarity))
        else:
            # Search across all clients
            for cid, client_embeddings in self.client_embeddings.items():
                for memory_id, client_embedding in client_embeddings.items():
                    similarity = np.dot(embedding, client_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(client_embedding)
                    )
                    similarities.append((cid, memory_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        # Get top k memories
        similar_memories = []
        for client_id, memory_id, similarity in similarities[:top_k]:
            memory = self.client_memories[client_id][memory_id].copy()
            memory['similarity'] = similarity
            similar_memories.append(memory)
        
        return similar_memories

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_memories': self.total_memories,
            'num_clients': len(self.client_memories),
            'memories_per_client': {
                client_id: len(memories)
                for client_id, memories in self.client_memories.items()
            },
            'privacy_budget_used': self.privacy_budget_used,
            'aggregation_rounds': len(self.aggregation_history)
        }

    def _train_local_model(
        self,
        client_id: int,
        epochs: int
    ) -> nn.Module:
        """Train a local model for a client."""
        local_model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        local_model.load_state_dict(self.global_model.state_dict())
        
        # Get client data
        embeddings = list(self.client_embeddings[client_id].values())
        if not embeddings:
            return local_model
            
        # Convert to tensors
        data = torch.FloatTensor(embeddings)
        
        # Train
        optimizer = torch.optim.Adam(local_model.parameters())
        criterion = nn.MSELoss()
        
        for _ in range(epochs):
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                output = local_model(batch)
                loss = criterion(output, batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return local_model

    def _aggregate_updates(self, local_updates: List[nn.Module]) -> None:
        """Aggregate local model updates with privacy."""
        # Get model parameters
        global_params = self.global_model.state_dict()
        
        # Average parameters with noise
        for key in global_params:
            param_sum = torch.zeros_like(global_params[key])
            for local_model in local_updates:
                param_sum += local_model.state_dict()[key]
            
            # Add noise to average
            avg_param = param_sum / len(local_updates)
            noisy_param = self.noise_generator.add_noise(avg_param.numpy())
            
            global_params[key] = torch.FloatTensor(noisy_param)
        
        # Update global model
        self.global_model.load_state_dict(global_params) 