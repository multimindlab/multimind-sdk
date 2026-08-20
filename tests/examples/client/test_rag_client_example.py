"""
Tests for rag_client_example.py client example.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.client.rag_client_example import example
from multimind.client.rag_client import Document, RAGClient


class MockResponse:
    """Mock aiohttp response for testing."""

    def __init__(self, status=200, json_data=None, text_data=None):
        self.status = status
        self._json_data = json_data or {}
        self._text_data = text_data or json.dumps(json_data or {})
        self.headers = {}

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class MockClientSession:
    """Mock aiohttp ClientSession for testing."""

    def __init__(self):
        self.responses = {}
        self.requests = []

    def set_response(self, url, method, response):
        """Set a mock response for a URL and method."""
        key = (method.upper(), url)
        self.responses[key] = response

    def post(self, url, **kwargs):
        """Mock POST request - returns context manager."""
        self.requests.append(("POST", url, kwargs))
        key = ("POST", url)
        if key in self.responses:
            return self.responses[key]
        return MockResponse(status=200, json_data={"status": "ok"})

    def get(self, url, **kwargs):
        """Mock GET request - returns context manager."""
        self.requests.append(("GET", url, kwargs))
        key = ("GET", url)
        if key in self.responses:
            return self.responses[key]
        return MockResponse(status=200, json_data={"status": "ok"})

    def delete(self, url, **kwargs):
        """Mock DELETE request - returns context manager."""
        self.requests.append(("DELETE", url, kwargs))
        key = ("DELETE", url)
        if key in self.responses:
            return self.responses[key]
        return MockResponse(status=200, json_data={"message": "deleted"})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


@pytest.fixture
def mock_client_session():
    """Fixture to create a mock client session."""
    return MockClientSession()


@pytest.mark.asyncio
async def test_rag_client_example_add_documents(mock_client_session):
    """Test adding documents via RAG client."""
    with patch('multimind.client.rag_client.aiohttp.ClientSession', new=lambda *args, **kwargs: mock_client_session):
        # Set up mock response
        mock_client_session.set_response(
            "http://localhost:8000/documents",
            "POST",
            MockResponse(
                status=200,
                json_data={
                    "documents": [
                        {
                            "text": "The RAG system provides powerful document processing.",
                            "metadata": {"type": "introduction"}
                        }
                    ],
                    "total": 1
                }
            )
        )

        client = RAGClient(base_url="http://localhost:8000")
        docs = [
            Document(
                text="The RAG system provides powerful document processing.",
                metadata={"type": "introduction"}
            )
        ]

        result = await client.add_documents(docs)

        assert result is not None
        assert "documents" in result
        assert result["total"] == 1
        assert len(mock_client_session.requests) == 1
        assert mock_client_session.requests[0][0] == "POST"
        assert "/documents" in mock_client_session.requests[0][1]


@pytest.mark.asyncio
async def test_rag_client_example_query(mock_client_session):
    """Test querying documents via RAG client."""
    with patch('multimind.client.rag_client.aiohttp.ClientSession', new=lambda *args, **kwargs: mock_client_session):
        # Set up mock response
        mock_client_session.set_response(
            "http://localhost:8000/query",
            "POST",
            MockResponse(
                status=200,
                json_data={
                    "documents": [
                        {
                            "text": "The RAG system provides powerful document processing.",
                            "metadata": {"type": "introduction"},
                            "score": 0.85
                        }
                    ],
                    "total": 1
                }
            )
        )

        client = RAGClient(base_url="http://localhost:8000")
        result = await client.query("What is the RAG system?", top_k=3)

        assert result is not None
        assert "documents" in result
        assert result["total"] == 1
        assert len(result["documents"]) == 1
        assert result["documents"][0]["score"] == 0.85
        assert len(mock_client_session.requests) == 1
        assert mock_client_session.requests[0][0] == "POST"
        assert "/query" in mock_client_session.requests[0][1]


@pytest.mark.asyncio
async def test_rag_client_example_generate(mock_client_session):
    """Test generating responses via RAG client."""
    with patch('multimind.client.rag_client.aiohttp.ClientSession', new=lambda *args, **kwargs: mock_client_session):
        # Set up mock response
        mock_client_session.set_response(
            "http://localhost:8000/generate",
            "POST",
            MockResponse(
                status=200,
                json_data={
                    "text": "The RAG (Retrieval Augmented Generation) system is a powerful document processing system that combines retrieval and generation capabilities.",
                    "documents": [
                        {
                            "text": "The RAG system provides powerful document processing.",
                            "metadata": {"type": "introduction"},
                            "score": 0.85
                        }
                    ]
                }
            )
        )

        client = RAGClient(base_url="http://localhost:8000")
        result = await client.generate(
            "Explain the RAG system",
            temperature=0.7
        )

        assert result is not None
        assert "text" in result
        assert "documents" in result
        assert len(result["documents"]) == 1
        assert "RAG" in result["text"]
        assert len(mock_client_session.requests) == 1
        assert mock_client_session.requests[0][0] == "POST"
        assert "/generate" in mock_client_session.requests[0][1]


@pytest.mark.asyncio
async def test_rag_client_example_get_document_count(mock_client_session):
    """Test getting document count via RAG client."""
    with patch('multimind.client.rag_client.aiohttp.ClientSession', new=lambda *args, **kwargs: mock_client_session):
        # Set up mock response
        mock_client_session.set_response(
            "http://localhost:8000/documents/count",
            "GET",
            MockResponse(
                status=200,
                json_data={"count": 5}
            )
        )

        client = RAGClient(base_url="http://localhost:8000")
        count = await client.get_document_count()

        assert count == 5
        assert len(mock_client_session.requests) == 1
        assert mock_client_session.requests[0][0] == "GET"
        assert "/documents/count" in mock_client_session.requests[0][1]


@pytest.mark.asyncio
async def test_rag_client_example_health_check(mock_client_session):
    """Test health check via RAG client."""
    with patch('multimind.client.rag_client.aiohttp.ClientSession', new=lambda *args, **kwargs: mock_client_session):
        # Set up mock response
        mock_client_session.set_response(
            "http://localhost:8000/health",
            "GET",
            MockResponse(
                status=200,
                json_data={
                    "status": "healthy",
                    "document_count": 5
                }
            )
        )

        client = RAGClient(base_url="http://localhost:8000")
        health = await client.health_check()

        assert health is not None
        assert health["status"] == "healthy"
        assert health["document_count"] == 5
        assert len(mock_client_session.requests) == 1
        assert mock_client_session.requests[0][0] == "GET"
        assert "/health" in mock_client_session.requests[0][1]


@pytest.mark.asyncio
async def test_rag_client_example_full_flow(mock_client_session):
    """Test the complete example flow."""
    with patch('multimind.client.rag_client.aiohttp.ClientSession', new=lambda *args, **kwargs: mock_client_session):
        # Set up all mock responses
        mock_client_session.set_response(
            "http://localhost:8000/documents",
            "POST",
            MockResponse(
                status=200,
                json_data={
                    "documents": [
                        {
                            "text": "The RAG system provides powerful document processing.",
                            "metadata": {"type": "introduction"}
                        }
                    ],
                    "total": 1
                }
            )
        )

        mock_client_session.set_response(
            "http://localhost:8000/query",
            "POST",
            MockResponse(
                status=200,
                json_data={
                    "documents": [
                        {
                            "text": "The RAG system provides powerful document processing.",
                            "metadata": {"type": "introduction"},
                            "score": 0.85
                        }
                    ],
                    "total": 1
                }
            )
        )

        mock_client_session.set_response(
            "http://localhost:8000/generate",
            "POST",
            MockResponse(
                status=200,
                json_data={
                    "text": "The RAG system is a document processing system.",
                    "documents": [
                        {
                            "text": "The RAG system provides powerful document processing.",
                            "metadata": {"type": "introduction"},
                            "score": 0.85
                        }
                    ]
                }
            )
        )

        mock_client_session.set_response(
            "http://localhost:8000/documents/count",
            "GET",
            MockResponse(status=200, json_data={"count": 1})
        )

        mock_client_session.set_response(
            "http://localhost:8000/health",
            "GET",
            MockResponse(
                status=200,
                json_data={
                    "status": "healthy",
                    "document_count": 1
                }
            )
        )

        # Run the example function
        with patch('builtins.print'):  # Suppress print output
            await example()

        # Verify all requests were made
        assert len(mock_client_session.requests) == 5

        # Check request order
        assert mock_client_session.requests[0][0] == "POST"
        assert "/documents" in mock_client_session.requests[0][1]

        assert mock_client_session.requests[1][0] == "POST"
        assert "/query" in mock_client_session.requests[1][1]

        assert mock_client_session.requests[2][0] == "POST"
        assert "/generate" in mock_client_session.requests[2][1]

        assert mock_client_session.requests[3][0] == "GET"
        assert "/documents/count" in mock_client_session.requests[3][1]

        assert mock_client_session.requests[4][0] == "GET"
        assert "/health" in mock_client_session.requests[4][1]


@pytest.mark.asyncio
async def test_rag_client_example_with_api_key(mock_client_session):
    """Test RAG client with API key authentication."""
    with patch('multimind.client.rag_client.aiohttp.ClientSession', new=lambda *args, **kwargs: mock_client_session):
        mock_client_session.set_response(
            "http://localhost:8000/documents",
            "POST",
            MockResponse(status=200, json_data={"documents": [], "total": 0})
        )

        client = RAGClient(
            base_url="http://localhost:8000",
            api_key="test-api-key"
        )

        assert "X-API-Key" in client.headers
        assert client.headers["X-API-Key"] == "test-api-key"

        docs = [Document(text="Test", metadata={})]
        await client.add_documents(docs)

        # Verify API key was sent in headers
        assert len(mock_client_session.requests) == 1
        request_kwargs = mock_client_session.requests[0][2]
        assert "headers" in request_kwargs
        assert request_kwargs["headers"]["X-API-Key"] == "test-api-key"


@pytest.mark.asyncio
async def test_rag_client_example_error_handling(mock_client_session):
    """Test error handling in RAG client."""
    with patch('multimind.client.rag_client.aiohttp.ClientSession', new=lambda *args, **kwargs: mock_client_session):
        # Set up error response
        mock_client_session.set_response(
            "http://localhost:8000/documents",
            "POST",
            MockResponse(
                status=500,
                json_data={"detail": "Internal server error"},
                text_data='{"detail": "Internal server error"}'
            )
        )

        client = RAGClient(base_url="http://localhost:8000")
        docs = [Document(text="Test", metadata={})]

        with pytest.raises(Exception) as exc_info:
            await client.add_documents(docs)

        assert "Failed to add documents" in str(exc_info.value) or "Internal server error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_rag_client_example_with_token(mock_client_session):
    """Test RAG client with JWT token authentication."""
    with patch('multimind.client.rag_client.aiohttp.ClientSession', new=lambda *args, **kwargs: mock_client_session):
        # Mock login response
        mock_client_session.set_response(
            "http://localhost:8000/token",
            "POST",
            MockResponse(
                status=200,
                json_data={
                    "access_token": "test-token-123",
                    "token_type": "bearer"
                }
            )
        )

        # Mock documents response
        mock_client_session.set_response(
            "http://localhost:8000/documents",
            "POST",
            MockResponse(status=200, json_data={"documents": [], "total": 0})
        )

        client = RAGClient(base_url="http://localhost:8000")

        # Login to get token
        token = await client.login("testuser", "secret")

        assert token == "test-token-123"
        assert "Authorization" in client.headers
        assert client.headers["Authorization"] == "Bearer test-token-123"

        # Use authenticated client
        docs = [Document(text="Test", metadata={})]
        await client.add_documents(docs)

        # Verify token was sent in headers
        assert len(mock_client_session.requests) == 2
        request_kwargs = mock_client_session.requests[1][2]
        assert "headers" in request_kwargs
        assert "Bearer test-token-123" in request_kwargs["headers"]["Authorization"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

