"""
Embedding standardizer for resizing and normalizing embeddings to match target dimensions.
"""

from typing import List

import numpy as np


class EmbeddingStandardizer:
    """Standardizes embeddings to match target dimensions."""

    def __init__(self):
        """Initialize the embedding standardizer."""
        pass

    def standardize(
        self, embedding: List[float], current_dimension: int, target_dimension: int
    ) -> List[float]:
        """
        Standardize an embedding to match the target dimension.

        Args:
            embedding: The embedding vector to standardize
            current_dimension: Current dimension of the embedding (can be inferred from embedding)
            target_dimension: Target dimension for the embedding

        Returns:
            Standardized embedding vector with target_dimension length
        """
        if not embedding:
            # Return zero vector if embedding is empty
            return [0.0] * target_dimension

        # Convert to numpy array for easier manipulation
        emb_array = np.array(embedding, dtype=np.float32)
        current_dim = len(emb_array)

        # Resize to target dimension
        if current_dim == target_dimension:
            # No resizing needed, just normalize
            standardized = emb_array
        elif current_dim > target_dimension:
            # Truncate if too long
            standardized = emb_array[:target_dimension]
        else:
            # Pad with zeros if too short
            padding = np.zeros(target_dimension - current_dim, dtype=np.float32)
            standardized = np.concatenate([emb_array, padding])

        # Normalize to unit vector
        norm = np.linalg.norm(standardized)
        if norm > 0:
            standardized = standardized / norm

        return standardized.tolist()
