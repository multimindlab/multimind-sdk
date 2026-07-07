"""
Base classes and interfaces for embedding generation.

DEPRECATION / DUPLICATION NOTE (module consolidation review): this module is
not imported by ``multimind.embeddings.__init__`` and is not referenced
anywhere else in the codebase — ``from multimind.embeddings import ...``
resolves to the canonical implementations in ``embedding.py`` (``Embedding``,
``EmbeddingType``) and ``embeddings.py`` (``EmbeddingConfig``,
``EmbeddingGenerator``). It is kept only for any external code that may
import ``multimind.embeddings.base`` directly.

- ``EmbeddingType`` below is a strict subset of the canonical
  ``embedding.EmbeddingType`` (identical string values for the members they
  share) and is now a re-export alias of it — genuinely redundant, so
  consolidated.
- ``EmbeddingConfig`` and ``EmbeddingGenerator`` are **not** aliased: they
  have different fields/semantics than their same-named siblings in
  ``embedding.py`` / ``embeddings.py`` (this ``EmbeddingConfig`` requires a
  ``dimension`` field the others don't have; this ``EmbeddingGenerator`` is a
  structural ``Protocol`` rather than the concrete dispatching class in
  ``embeddings.py``). Aliasing those would be a behavior change, so they are
  left as-is and documented here instead.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .embedding import EmbeddingType  # noqa: F401 - re-export, see note above


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation.

    Not the canonical ``EmbeddingConfig`` (see module docstring) — kept
    separate because it has a ``dimension`` field the canonical configs
    (in ``embedding.py`` / ``embeddings.py``) do not.
    """

    model_name: str  # Name of the embedding model
    dimension: int  # Dimension of the embeddings
    batch_size: int = 32  # Batch size for generation
    device: str = "cpu"  # Device to use for generation
    custom_params: Dict[str, Any] = None  # Custom parameters


@runtime_checkable
class EmbeddingGenerator(Protocol):
    """Protocol defining embedding generator interface.

    Not the canonical ``EmbeddingGenerator`` (see module docstring) — this is
    a structural typing ``Protocol`` with no implementation, whereas
    ``embeddings.EmbeddingGenerator`` (the one exported by
    ``multimind.embeddings``) is a concrete class that dispatches to a
    provider-specific embedder based on ``EmbeddingConfig.model_name``.
    """

    async def initialize(self) -> None:
        """Initialize the embedding generator."""
        pass

    async def generate(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass

    async def generate_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @property
    def dimension(self) -> int:
        """Get the dimension of the embeddings."""
        pass
