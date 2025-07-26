import pytest
from multimind.core.exceptions import (
    MultiMindError,
    RetrievalError,
    GenerationError,
    DocumentProcessingError,
    EmbeddingError,
    ConfigurationError
)

def test_multimind_error():
    with pytest.raises(MultiMindError):
        raise MultiMindError("test error")

def test_retrieval_error():
    with pytest.raises(RetrievalError):
        raise RetrievalError("test error")

def test_generation_error():
    with pytest.raises(GenerationError):
        raise GenerationError("test error")

def test_document_processing_error():
    with pytest.raises(DocumentProcessingError):
        raise DocumentProcessingError("test error")

def test_embedding_error():
    with pytest.raises(EmbeddingError):
        raise EmbeddingError("test error")

def test_configuration_error():
    with pytest.raises(ConfigurationError):
        raise ConfigurationError("test error") 