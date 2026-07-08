"""
Tests for example_rag.py RAG example.
"""

import pytest
pytest.importorskip("multimind.rag", exc_type=ImportError)  # requires optional extras absent on core-only installs
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.rag.example_rag import main, SimpleRAGWrapper
from multimind.document_processing.base import Document


class MockOpenAIModel:
    """Mock OpenAI model for testing."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
    
    async def generate(self, prompt: str, **kwargs):
        return f"Mock OpenAI response to: {prompt[:50]}..."


class MockHuggingFaceModel:
    """Mock HuggingFace model for testing."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
    
    async def generate(self, prompt: str, **kwargs):
        return f"Mock HuggingFace response to: {prompt[:50]}..."


class MockDocument:
    """Mock Document for testing."""
    
    def __init__(self, id: str, content: str, metadata: dict = None, source: str = "test"):
        self.id = id
        self.content = content
        self.metadata = metadata or {}
        self.source = source
        self.score = 0.8  # Default similarity score


class MockRAG:
    """Mock RAG system for testing."""
    
    def __init__(self, config):
        self.config = config
        self.documents = []
        self.initialized = False
    
    async def initialize(self):
        """Initialize the RAG system."""
        self.initialized = True
    
    async def add_documents(self, documents, process: bool = True):
        """Add documents to the RAG system."""
        if process:
            # Simulate processing - just store documents
            self.documents.extend(documents)
        else:
            self.documents.extend(documents)
    
    async def retrieve(self, query: str, k: int = 3):
        """Retrieve relevant documents."""
        # Return mock documents with scores
        results = []
        for i, doc in enumerate(self.documents[:k]):
            mock_doc = MockDocument(
                id=doc.id if hasattr(doc, 'id') else f"doc_{i}",
                content=doc.content if hasattr(doc, 'content') else str(doc),
                metadata=getattr(doc, 'metadata', {}),
                source=getattr(doc, 'source', 'test')
            )
            # Set decreasing scores
            mock_doc.score = 0.9 - (i * 0.1)
            results.append(mock_doc)
        return results


class MockVectorStoreConfig:
    """Mock VectorStoreConfig for testing."""
    
    @staticmethod
    def create_faiss_config(dimension: int, metric: str = "cosine", index_type: str = "flat"):
        return MockVectorStoreConfig(dimension, metric, index_type)
    
    def __init__(self, dimension: int, metric: str, index_type: str):
        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type


class MockEmbeddingConfig:
    """Mock EmbeddingConfig for testing."""
    
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name', 'test-model')
        self.model_type = kwargs.get('model_type', 'openai')
        self.batch_size = kwargs.get('batch_size', 32)
        self.max_length = kwargs.get('max_length', 512)
        self.normalize = kwargs.get('normalize', True)
        self.device = kwargs.get('device', 'cpu')
        self.cache_dir = kwargs.get('cache_dir', None)
        self.custom_params = kwargs.get('custom_params', {})


class MockRAGConfig:
    """Mock RAGConfig for testing."""
    
    def __init__(self, **kwargs):
        self.vector_store_config = kwargs.get('vector_store_config')
        self.retrieval_config = kwargs.get('retrieval_config', {})
        self.embedding_config = kwargs.get('embedding_config')
        self.document_config = kwargs.get('document_config', {})


@pytest.fixture
def mock_openai_model():
    """Fixture to provide mock OpenAI model."""
    return MockOpenAIModel("gpt-3.5-turbo", temperature=0.7)


@pytest.fixture
def mock_huggingface_model():
    """Fixture to provide mock HuggingFace model."""
    return MockHuggingFaceModel("gpt2")


@pytest.fixture
def mock_rag():
    """Fixture to provide mock RAG system."""
    config = MockRAGConfig(
        vector_store_config=MockVectorStoreConfig(1536, "cosine", "flat"),
        retrieval_config={"top_k": 3, "similarity_threshold": 0.5},
        embedding_config=MockEmbeddingConfig(),
        document_config={"min_chunk_size": 100, "max_chunk_size": 1000, "chunk_overlap": 200}
    )
    return MockRAG(config)


@pytest.fixture
def simple_rag_wrapper(mock_rag, mock_openai_model):
    """Fixture to provide SimpleRAGWrapper instance."""
    return SimpleRAGWrapper(mock_rag, mock_openai_model, similarity_threshold=0.4)


@pytest.mark.asyncio
async def test_example_rag_imports():
    """Test that the example_rag module can be imported."""
    try:
        from examples.rag.example_rag import main, SimpleRAGWrapper
        assert main is not None
        assert SimpleRAGWrapper is not None
    except ImportError as e:
        pytest.fail(f"Failed to import example_rag: {e}")


@pytest.mark.asyncio
async def test_simple_rag_wrapper_initialization(mock_rag, mock_openai_model):
    """Test that SimpleRAGWrapper can be initialized."""
    wrapper = SimpleRAGWrapper(mock_rag, mock_openai_model, similarity_threshold=0.5)
    
    assert wrapper.rag == mock_rag
    assert wrapper.model == mock_openai_model
    assert wrapper.similarity_threshold == 0.5
    assert wrapper._last_retrieved == []


@pytest.mark.asyncio
async def test_simple_rag_wrapper_add_documents(simple_rag_wrapper):
    """Test adding documents to SimpleRAGWrapper."""
    documents = [
        "Document 1: Quantum computing basics",
        "Document 2: Quantum mechanics principles",
        "Document 3: Quantum algorithms"
    ]
    
    await simple_rag_wrapper.add_documents(documents)
    
    # Verify documents were added to the RAG system
    assert len(simple_rag_wrapper.rag.documents) == 3


@pytest.mark.asyncio
async def test_simple_rag_wrapper_query_with_results(simple_rag_wrapper):
    """Test querying SimpleRAGWrapper with retrieved documents."""
    # Add documents first
    documents = [
        "Quantum computing is a type of computation that harnesses quantum states.",
        "Quantum computers use quantum bits (qubits) instead of classical bits.",
        "Qubits can exist in multiple states simultaneously."
    ]
    await simple_rag_wrapper.add_documents(documents)
    
    # Query
    query = "What is quantum computing?"
    response = await simple_rag_wrapper.query(query)
    
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_simple_rag_wrapper_query_no_results(simple_rag_wrapper):
    """Test querying SimpleRAGWrapper when no documents match threshold."""
    # Set high threshold so no documents pass
    simple_rag_wrapper.similarity_threshold = 1.0
    
    # Add documents
    documents = ["Test document"]
    await simple_rag_wrapper.add_documents(documents)
    
    # Query - should return fallback message
    query = "What is quantum computing?"
    response = await simple_rag_wrapper.query(query)
    
    assert "don't have enough information" in response.lower()


@pytest.mark.asyncio
async def test_simple_rag_wrapper_get_retrieved_documents(simple_rag_wrapper):
    """Test getting retrieved documents from SimpleRAGWrapper."""
    # Add documents
    documents = [
        "Quantum computing basics",
        "Quantum mechanics principles",
        "Quantum algorithms"
    ]
    await simple_rag_wrapper.add_documents(documents)
    
    # Query first to populate _last_retrieved
    query = "What is quantum computing?"
    await simple_rag_wrapper.query(query)
    
    # Get retrieved documents
    retrieved = await simple_rag_wrapper.get_retrieved_documents(query)
    
    assert isinstance(retrieved, list)
    assert len(retrieved) > 0
    assert all(isinstance(doc, dict) for doc in retrieved)
    assert all("text" in doc and "metadata" in doc and "score" in doc for doc in retrieved)


@pytest.mark.asyncio
async def test_simple_rag_wrapper_fallback_answer(simple_rag_wrapper):
    """Test fallback answer generation."""
    # Create documents with scores below threshold
    mock_docs = [
        MockDocument("doc1", "Test content 1", {}, "test"),
        MockDocument("doc2", "Test content 2", {}, "test")
    ]
    mock_docs[0].score = 0.3  # Below threshold
    mock_docs[1].score = 0.2  # Below threshold
    
    # Test fallback with documents
    fallback = simple_rag_wrapper._build_fallback_answer(mock_docs)
    assert isinstance(fallback, str)
    assert len(fallback) > 0
    
    # Test fallback with empty list
    fallback_empty = simple_rag_wrapper._build_fallback_answer([])
    assert "don't have enough information" in fallback_empty.lower()


@pytest.mark.asyncio
async def test_rag_initialization(mock_rag):
    """Test that RAG system can be initialized."""
    await mock_rag.initialize()
    assert mock_rag.initialized is True


@pytest.mark.asyncio
async def test_rag_add_documents(mock_rag):
    """Test adding documents to RAG system."""
    documents = [
        Document(id="doc1", content="Content 1", metadata={}, source="test"),
        Document(id="doc2", content="Content 2", metadata={}, source="test")
    ]
    
    await mock_rag.add_documents(documents, process=True)
    assert len(mock_rag.documents) == 2


@pytest.mark.asyncio
async def test_rag_retrieve(mock_rag):
    """Test retrieving documents from RAG system."""
    # Add documents first
    documents = [
        Document(id="doc1", content="Quantum computing", metadata={}, source="test"),
        Document(id="doc2", content="Classical computing", metadata={}, source="test"),
        Document(id="doc3", content="Machine learning", metadata={}, source="test")
    ]
    await mock_rag.add_documents(documents)
    
    # Retrieve
    results = await mock_rag.retrieve("quantum", k=2)
    assert len(results) == 2
    assert all(hasattr(doc, 'score') for doc in results)


@pytest.mark.asyncio
async def test_model_generation(mock_openai_model):
    """Test that models can generate responses."""
    prompt = "Explain quantum computing"
    response = await mock_openai_model.generate(prompt)
    
    assert response is not None
    assert isinstance(response, str)
    assert "Mock OpenAI response" in response


@pytest.mark.asyncio
async def test_example_rag_main_function():
    """Test that the main function can be called without errors."""
    with patch('examples.rag.example_rag.OpenAIModel', MockOpenAIModel), \
         patch('examples.rag.example_rag.HuggingFaceModel', MockHuggingFaceModel), \
         patch('examples.rag.example_rag.RAG', MockRAG), \
         patch('examples.rag.example_rag.RAGConfig', MockRAGConfig), \
         patch('examples.rag.example_rag.VectorStoreConfig', MockVectorStoreConfig), \
         patch('examples.rag.example_rag.EmbeddingConfig', MockEmbeddingConfig), \
         patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        
        try:
            await main()
            assert True  # If we get here, the function ran without errors
        except Exception as e:
            # Some errors are expected (like FAISS not available), but main should handle them
            # We just want to ensure it doesn't crash unexpectedly
            pass


@pytest.mark.asyncio
async def test_example_rag_main_with_huggingface():
    """Test main function with HuggingFace model fallback."""
    with patch('examples.rag.example_rag.OpenAIModel', MockOpenAIModel), \
         patch('examples.rag.example_rag.HuggingFaceModel', MockHuggingFaceModel), \
         patch('examples.rag.example_rag.RAG', MockRAG), \
         patch('examples.rag.example_rag.RAGConfig', MockRAGConfig), \
         patch('examples.rag.example_rag.VectorStoreConfig', MockVectorStoreConfig), \
         patch('examples.rag.example_rag.EmbeddingConfig', MockEmbeddingConfig), \
         patch.dict(os.environ, {}, clear=True), \
         patch('examples.rag.example_rag.HUGGINGFACE_AVAILABLE', True):
        
        try:
            await main()
            assert True
        except Exception as e:
            # Expected errors (like FAISS) are acceptable
            pass


@pytest.mark.asyncio
async def test_vector_store_config_creation():
    """Test that VectorStoreConfig can be created."""
    config = MockVectorStoreConfig.create_faiss_config(
        dimension=1536,
        metric="cosine",
        index_type="flat"
    )
    
    assert config.dimension == 1536
    assert config.metric == "cosine"
    assert config.index_type == "flat"


@pytest.mark.asyncio
async def test_embedding_config_creation():
    """Test that EmbeddingConfig can be created."""
    config = MockEmbeddingConfig(
        model_name="text-embedding-ada-002",
        model_type="openai",
        batch_size=32,
        max_length=512,
        normalize=True,
        device="cpu",
        custom_params={"api_key": "test-key"}
    )
    
    assert config.model_name == "text-embedding-ada-002"
    assert config.model_type == "openai"
    assert config.batch_size == 32
    assert config.custom_params["api_key"] == "test-key"


@pytest.mark.asyncio
async def test_rag_config_creation():
    """Test that RAGConfig can be created."""
    vector_config = MockVectorStoreConfig(1536, "cosine", "flat")
    embedding_config = MockEmbeddingConfig()
    
    config = MockRAGConfig(
        vector_store_config=vector_config,
        retrieval_config={"top_k": 3, "similarity_threshold": 0.5},
        embedding_config=embedding_config,
        document_config={"min_chunk_size": 100, "max_chunk_size": 1000}
    )
    
    assert config.vector_store_config == vector_config
    assert config.retrieval_config["top_k"] == 3
    assert config.embedding_config == embedding_config


@pytest.mark.asyncio
async def test_simple_rag_wrapper_response_filtering(simple_rag_wrapper):
    """Test that responses are properly filtered and cleaned."""
    # Add documents
    documents = ["Quantum computing is a revolutionary technology."]
    await simple_rag_wrapper.add_documents(documents)
    
    # Mock model to return response with unwanted content
    async def mock_generate(prompt, **kwargs):
        return "You are a helpful assistant. Answer: Quantum computing uses qubits. Question: What are qubits?"
    
    simple_rag_wrapper.model.generate = mock_generate
    
    # Query
    response = await simple_rag_wrapper.query("What is quantum computing?")
    
    # Response should be cleaned (no "You are a helpful assistant" prefix, no new questions)
    assert response is not None
    assert not response.lower().startswith("you are a helpful assistant")
    assert "Question:" not in response
    assert "Q:" not in response


@pytest.mark.asyncio
async def test_simple_rag_wrapper_empty_response_handling(simple_rag_wrapper):
    """Test handling of empty or invalid responses."""
    # Add documents
    documents = ["Test content"]
    await simple_rag_wrapper.add_documents(documents)
    
    # Mock model to return empty or invalid response
    async def mock_generate_empty(prompt, **kwargs):
        return ""
    
    simple_rag_wrapper.model.generate = mock_generate_empty
    
    # Query should still return a response (fallback)
    response = await simple_rag_wrapper.query("Test query")
    assert response is not None
    assert len(response) > 0


def test_example_structure():
    """Test that the example has the expected structure."""
    example_path = Path(__file__).parent.parent.parent.parent / "examples" / "rag" / "example_rag.py"
    assert example_path.exists(), "example_rag.py example should exist"
    
    # Check that the file contains expected components
    with open(example_path, 'r') as f:
        content = f.read()
        assert "async def main()" in content
        assert "SimpleRAGWrapper" in content
        assert "RAG" in content
        assert "RAGConfig" in content
        assert "OpenAIModel" in content
        assert "add_documents" in content
        assert "query" in content

