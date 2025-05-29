"""
Neuro-Symbolic Hybrid Memory implementation combining neural embeddings with symbolic reasoning.
"""

from typing import Dict, Any, Optional, List, Set, Tuple
import numpy as np
from datetime import datetime
import networkx as nx
from .base import BaseMemory
from .vector_store import VectorStoreMemory
from .knowledge_graph import KnowledgeGraphMemory

class NeuroSymbolicMemory(BaseMemory):
    """Memory implementation combining neural embeddings with symbolic reasoning."""

    def __init__(
        self,
        embedding_dim: int = 768,
        similarity_threshold: float = 0.7,
        symbolic_weight: float = 0.5,
        neural_weight: float = 0.5,
        **kwargs
    ):
        """Initialize neuro-symbolic memory."""
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.symbolic_weight = symbolic_weight
        self.neural_weight = neural_weight
        
        # Neural component using vector store
        self.neural_memory = VectorStoreMemory(
            embedding_dim=embedding_dim,
            similarity_threshold=similarity_threshold
        )
        
        # Symbolic component using knowledge graph
        self.symbolic_memory = KnowledgeGraphMemory()
        
        # Cross-modal mapping
        self.neural_to_symbolic: Dict[str, str] = {}
        self.symbolic_to_neural: Dict[str, str] = {}
        
        # Confidence scores for mappings
        self.mapping_confidence: Dict[Tuple[str, str], float] = {}

    async def add(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        symbolic_relations: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add a memory entry with both neural and symbolic representations."""
        # Add to neural memory
        neural_key = f"neural_{key}"
        await self.neural_memory.add(neural_key, value, metadata)
        
        # Add to symbolic memory if relations provided
        if symbolic_relations:
            symbolic_key = f"symbolic_{key}"
            await self.symbolic_memory.add(symbolic_key, value, metadata)
            
            # Add relations to knowledge graph
            for relation in symbolic_relations:
                await self.symbolic_memory.add_relation(
                    symbolic_key,
                    relation['target'],
                    relation['type'],
                    relation.get('metadata')
                )
            
            # Create cross-modal mapping
            self.neural_to_symbolic[neural_key] = symbolic_key
            self.symbolic_to_neural[symbolic_key] = neural_key
            self.mapping_confidence[(neural_key, symbolic_key)] = 1.0

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a memory entry using both neural and symbolic components."""
        # Try neural retrieval first
        neural_key = f"neural_{key}"
        neural_result = await self.neural_memory.get(neural_key)
        
        # Try symbolic retrieval if available
        symbolic_key = f"symbolic_{key}"
        symbolic_result = await self.symbolic_memory.get(symbolic_key)
        
        # Combine results based on weights
        if neural_result and symbolic_result:
            return {
                'neural': neural_result,
                'symbolic': symbolic_result,
                'confidence': self.mapping_confidence.get((neural_key, symbolic_key), 0.5)
            }
        elif neural_result:
            return {'neural': neural_result, 'confidence': 1.0}
        elif symbolic_result:
            return {'symbolic': symbolic_result, 'confidence': 1.0}
        return None

    async def find_similar(
        self,
        query: str,
        top_k: int = 5,
        use_neural: bool = True,
        use_symbolic: bool = True
    ) -> List[Dict[str, Any]]:
        """Find similar memories using both neural and symbolic components."""
        results = []
        
        if use_neural:
            neural_results = await self.neural_memory.find_similar(query, top_k)
            for result in neural_results:
                neural_key = result['key']
                symbolic_key = self.neural_to_symbolic.get(neural_key)
                confidence = self.mapping_confidence.get((neural_key, symbolic_key), 0.5)
                
                results.append({
                    'neural': result,
                    'symbolic_key': symbolic_key,
                    'confidence': confidence * self.neural_weight
                })
        
        if use_symbolic:
            symbolic_results = await self.symbolic_memory.find_similar(query, top_k)
            for result in symbolic_results:
                symbolic_key = result['key']
                neural_key = self.symbolic_to_neural.get(symbolic_key)
                confidence = self.mapping_confidence.get((neural_key, symbolic_key), 0.5)
                
                results.append({
                    'symbolic': result,
                    'neural_key': neural_key,
                    'confidence': confidence * self.symbolic_weight
                })
        
        # Sort by combined confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:top_k]

    async def get_relations(
        self,
        key: str,
        relation_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get symbolic relations for a memory entry."""
        symbolic_key = f"symbolic_{key}"
        return await self.symbolic_memory.get_relations(symbolic_key, relation_type)

    async def update_mapping_confidence(
        self,
        neural_key: str,
        symbolic_key: str,
        new_confidence: float
    ) -> None:
        """Update the confidence score of a cross-modal mapping."""
        self.mapping_confidence[(neural_key, symbolic_key)] = new_confidence

    async def get_memory_stats(self, key: str) -> Dict[str, Any]:
        """Get detailed statistics for a memory entry."""
        neural_key = f"neural_{key}"
        symbolic_key = f"symbolic_{key}"
        
        neural_stats = await self.neural_memory.get_stats()
        symbolic_stats = await self.symbolic_memory.get_stats()
        
        return {
            'neural_stats': neural_stats,
            'symbolic_stats': symbolic_stats,
            'mapping_confidence': self.mapping_confidence.get((neural_key, symbolic_key), 0.0),
            'has_neural': neural_key in self.neural_to_symbolic,
            'has_symbolic': symbolic_key in self.symbolic_to_neural
        }

    async def remove(self, key: str) -> None:
        """Remove a memory entry from both neural and symbolic components."""
        neural_key = f"neural_{key}"
        symbolic_key = f"symbolic_{key}"
        
        # Remove from neural memory
        await self.neural_memory.remove(neural_key)
        
        # Remove from symbolic memory
        await self.symbolic_memory.remove(symbolic_key)
        
        # Remove mappings
        if neural_key in self.neural_to_symbolic:
            del self.neural_to_symbolic[neural_key]
        if symbolic_key in self.symbolic_to_neural:
            del self.symbolic_to_neural[symbolic_key]
        if (neural_key, symbolic_key) in self.mapping_confidence:
            del self.mapping_confidence[(neural_key, symbolic_key)]

    async def clear(self) -> None:
        """Clear all memory entries and mappings."""
        await self.neural_memory.clear()
        await self.symbolic_memory.clear()
        self.neural_to_symbolic.clear()
        self.symbolic_to_neural.clear()
        self.mapping_confidence.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """Get overall memory statistics."""
        neural_stats = await self.neural_memory.get_stats()
        symbolic_stats = await self.symbolic_memory.get_stats()
        
        return {
            'neural_stats': neural_stats,
            'symbolic_stats': symbolic_stats,
            'total_mappings': len(self.mapping_confidence),
            'avg_mapping_confidence': sum(self.mapping_confidence.values()) / max(1, len(self.mapping_confidence)),
            'neural_weight': self.neural_weight,
            'symbolic_weight': self.symbolic_weight
        } 