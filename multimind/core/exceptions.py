"""
Common exceptions for the MultiMind SDK.
"""

class MultiMindError(Exception):
    """Base exception for MultiMind SDK."""
    pass

class RetrievalError(MultiMindError):
    """Raised when there's an error during retrieval."""
    pass

class GenerationError(MultiMindError):
    """Raised when there's an error during generation."""
    pass

class DocumentProcessingError(MultiMindError):
    """Raised when there's an error processing documents."""
    pass

class VectorStoreError(MultiMindError):
    """Raised when there's an error with vector store operations."""
    pass

class EmbeddingError(MultiMindError):
    """Raised when there's an error with embedding operations."""
    pass

class ConfigurationError(MultiMindError):
    """Raised when there's a configuration error."""
    pass

class ValidationError(MultiMindError):
    """Raised when there's a validation error."""
    pass 