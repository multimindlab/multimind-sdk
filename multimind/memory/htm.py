"""
Hierarchical Temporal Memory (HTM) implementation.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
from torch import nn
from .base import BaseMemory

class SparseDistributedRepresentation:
    """Sparse distributed representation for HTM."""
    def __init__(self, size: int, sparsity: float = 0.02):
        self.size = size
        self.sparsity = sparsity
        self.active_bits = set()
        
    def encode(self, data: np.ndarray) -> None:
        """Encode data into sparse representation."""
        # Sort values and take top k
        k = int(self.size * self.sparsity)
        top_k_idx = np.argsort(data)[-k:]
        self.active_bits = set(top_k_idx)
        
    def overlap(self, other: 'SparseDistributedRepresentation') -> float:
        """Calculate overlap with another SDR."""
        return len(self.active_bits.intersection(other.active_bits)) / len(self.active_bits)

class HTMColumn:
    """HTM column with cells and synapses."""
    def __init__(self, num_cells: int = 4):
        self.num_cells = num_cells
        self.cells = [False] * num_cells  # Active state
        self.predictive_cells = [False] * num_cells
        self.synapses = {}  # (column_idx, cell_idx) -> permanence
        
    def update(self, active: bool) -> None:
        """Update column state."""
        if active:
            # Activate all cells
            self.cells = [True] * self.num_cells
        else:
            # Only predictive cells remain active
            self.cells = self.predictive_cells.copy()

class HTMMemory(BaseMemory):
    """Implements Hierarchical Temporal Memory."""
    
    def __init__(
        self,
        input_size: int = 1024,
        num_columns: int = 2048,
        cells_per_column: int = 4,
        sparsity: float = 0.02,
        learning_rate: float = 0.1,
        **kwargs
    ):
        """Initialize HTM memory."""
        super().__init__(**kwargs)
        
        # HTM parameters
        self.input_size = input_size
        self.num_columns = num_columns
        self.cells_per_column = cells_per_column
        self.sparsity = sparsity
        self.learning_rate = learning_rate
        
        # Initialize HTM components
        self.columns = [HTMColumn(cells_per_column) for _ in range(num_columns)]
        self.input_sdr = SparseDistributedRepresentation(input_size, sparsity)
        self.memory_sdr = SparseDistributedRepresentation(num_columns, sparsity)
        
        # Memory tracking
        self.sequence_memories: List[List[int]] = []
        self.anomaly_scores: List[float] = []
        
        # Statistics
        self.total_sequences = 0
        self.total_predictions = 0
        self.avg_anomaly_score = 0.0

    async def add_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add memory to HTM."""
        # Convert content to input representation
        input_data = self._get_input_representation(content)
        
        # Encode input
        self.input_sdr.encode(input_data)
        
        # Update columns
        active_columns = self._update_columns()
        
        # Update memory SDR
        self.memory_sdr.encode(active_columns)
        
        # Store sequence
        self.sequence_memories.append(active_columns)
        self.total_sequences += 1
        
        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(active_columns)
        self.anomaly_scores.append(anomaly_score)
        self.avg_anomaly_score = (
            self.avg_anomaly_score * (self.total_sequences - 1) +
            anomaly_score
        ) / self.total_sequences

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory using HTM prediction."""
        # Convert query to input representation
        query_data = self._get_input_representation(memory_id)
        
        # Encode query
        self.input_sdr.encode(query_data)
        
        # Get predictions
        predicted_columns = self._get_predictions()
        self.total_predictions += 1
        
        if predicted_columns:
            # Find most similar sequence
            best_sequence = self._find_best_sequence(predicted_columns)
            
            if best_sequence:
                return {
                    'id': memory_id,
                    'content': self._decode_sequence(best_sequence),
                    'prediction_confidence': self._calculate_confidence(predicted_columns),
                    'anomaly_score': self.anomaly_scores[-1]
                }
        return None

    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update memory in HTM."""
        if 'content' in updates:
            # Convert new content to input representation
            new_data = self._get_input_representation(updates['content'])
            
            # Encode new input
            self.input_sdr.encode(new_data)
            
            # Update columns
            active_columns = self._update_columns()
            
            # Update memory SDR
            self.memory_sdr.encode(active_columns)
            
            # Update sequence
            if self.sequence_memories:
                self.sequence_memories[-1] = active_columns

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_sequences': self.total_sequences,
            'total_predictions': self.total_predictions,
            'avg_anomaly_score': self.avg_anomaly_score,
            'active_columns': len([c for c in self.columns if any(c.cells)]),
            'predictive_columns': len([c for c in self.columns if any(c.predictive_cells)])
        }

    def _update_columns(self) -> List[int]:
        """Update HTM columns based on input."""
        active_columns = []
        
        for i, column in enumerate(self.columns):
            # Check if column should be active
            if self._should_activate_column(i):
                column.update(True)
                active_columns.append(i)
            else:
                column.update(False)
                
            # Update synapses
            self._update_synapses(i)
            
        return active_columns

    def _should_activate_column(self, column_idx: int) -> bool:
        """Determine if a column should be active."""
        # This would typically use more sophisticated activation rules
        # For now, we'll use a simple threshold
        return np.random.random() < 0.1

    def _update_synapses(self, column_idx: int) -> None:
        """Update synapses for a column."""
        # This would typically implement Hebbian learning
        # For now, we'll use a simple random update
        for i in range(self.num_columns):
            if (column_idx, i) not in self.columns[column_idx].synapses:
                self.columns[column_idx].synapses[(column_idx, i)] = np.random.random()

    def _get_predictions(self) -> List[int]:
        """Get predictions from HTM."""
        predicted_columns = []
        
        for i, column in enumerate(self.columns):
            if any(column.predictive_cells):
                predicted_columns.append(i)
                
        return predicted_columns

    def _find_best_sequence(self, predicted_columns: List[int]) -> Optional[List[int]]:
        """Find best matching sequence."""
        if not self.sequence_memories:
            return None
            
        best_sequence = None
        best_overlap = 0.0
        
        for sequence in self.sequence_memories:
            overlap = len(set(predicted_columns).intersection(set(sequence))) / len(predicted_columns)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sequence = sequence
                
        return best_sequence if best_overlap > 0.5 else None

    def _calculate_anomaly_score(self, active_columns: List[int]) -> float:
        """Calculate anomaly score for current input."""
        if not self.sequence_memories:
            return 1.0
            
        # Calculate average overlap with past sequences
        overlaps = [
            len(set(active_columns).intersection(set(seq))) / len(active_columns)
            for seq in self.sequence_memories
        ]
        
        return 1.0 - np.mean(overlaps)

    def _calculate_confidence(self, predicted_columns: List[int]) -> float:
        """Calculate prediction confidence."""
        return len(predicted_columns) / self.num_columns

    def _get_input_representation(self, text: str) -> np.ndarray:
        """Convert text to input representation."""
        # This would typically use a more sophisticated encoding
        # For now, we'll use a simple random projection
        return np.random.randn(self.input_size)

    def _decode_sequence(self, sequence: List[int]) -> str:
        """Convert sequence back to text."""
        # This would typically use a decoder model
        # For now, we'll return a placeholder
        return f"Memory sequence with {len(sequence)} active columns" 