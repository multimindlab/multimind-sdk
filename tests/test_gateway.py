"""
Tests for the MultiMind Gateway module
"""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from multimind.gateway.api import app, ModelResponse
from multimind.gateway.cli import MultiMindCLI
from multimind.gateway.models import get_model_handler
from multimind.gateway.config import config

# Test client for FastAPI
client = TestClient(app)

# Mock responses
MOCK_RESPONSE = ModelResponse(
    content="This is a test response",
    model="test-model",
    usage={"prompt_tokens": 10, "completion_tokens": 20},
    finish_reason="stop"
)

@pytest.fixture
def mock_model_handler():
    """Fixture to mock model handler"""
    with patch("multimind.gateway.models.get_model_handler") as mock:
        handler = MagicMock()
        handler.chat.return_value = MOCK_RESPONSE
        handler.generate.return_value = MOCK_RESPONSE
        mock.return_value = handler
        yield mock

@pytest.fixture
def mock_config():
    """Fixture to mock configuration"""
    with patch("multimind.gateway.config.config") as mock:
        mock.validate.return_value = {
            "openai": True,
            "anthropic": True,
            "ollama": True,
            "groq": True,
            "huggingface": True
        }
        yield mock

def test_api_root():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "name" in response.json()
    assert "version" in response.json()
    assert "models" in response.json()

def test_api_list_models(mock_config):
    """Test the models endpoint"""
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert "models" in response.json()
    assert all(model in response.json()["models"] for model in ["openai", "anthropic", "ollama"])

def test_api_chat(mock_model_handler, mock_config):
    """Test the chat endpoint"""
    request_data = {
        "messages": [
            {"role": "user", "content": "Hello, world!"}
        ],
        "model": "openai",
        "temperature": 0.7
    }
    response = client.post("/v1/chat", json=request_data)
    assert response.status_code == 200
    assert response.json()["content"] == MOCK_RESPONSE.content
    assert response.json()["model"] == MOCK_RESPONSE.model

def test_api_generate(mock_model_handler, mock_config):
    """Test the generate endpoint"""
    request_data = {
        "prompt": "Hello, world!",
        "model": "openai",
        "temperature": 0.7
    }
    response = client.post("/v1/generate", json=request_data)
    assert response.status_code == 200
    assert response.json()["content"] == MOCK_RESPONSE.content
    assert response.json()["model"] == MOCK_RESPONSE.model

def test_api_compare(mock_model_handler, mock_config):
    """Test the compare endpoint"""
    request_data = {
        "prompt": "Hello, world!",
        "models": ["openai", "anthropic"]
    }
    response = client.post("/v1/compare", json=request_data)
    assert response.status_code == 200
    assert "responses" in response.json()
    assert all(model in response.json()["responses"] for model in ["openai", "anthropic"])

@pytest.mark.asyncio
async def test_cli_chat(mock_model_handler, mock_config):
    """Test the CLI chat functionality"""
    cli = MultiMindCLI()
    cli.validate_config()  # Should not raise any exceptions
    
    # Test single message mode
    await cli.chat("openai", "Hello, world!")
    mock_model_handler.assert_called_once_with("openai")
    
    # Test interactive mode (simulated)
    with patch("click.prompt") as mock_prompt:
        mock_prompt.side_effect = ["Hello", "exit"]
        await cli.chat("openai")
        assert mock_prompt.call_count == 2

@pytest.mark.asyncio
async def test_cli_compare(mock_model_handler, mock_config):
    """Test the CLI compare functionality"""
    cli = MultiMindCLI()
    cli.validate_config()
    
    await cli.compare("Hello, world!", ["openai", "anthropic"])
    assert mock_model_handler.call_count == 2

def test_model_handler_factory():
    """Test the model handler factory function"""
    with pytest.raises(ValueError):
        get_model_handler("invalid-model")
    
    # Test with mock config
    with patch("multimind.gateway.config.config") as mock_config:
        mock_config.get_model_config.return_value = MagicMock()
        handler = get_model_handler("openai")
        assert handler is not None

def test_error_handling():
    """Test error handling in API endpoints"""
    # Test with invalid model
    request_data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "invalid-model"
    }
    response = client.post("/v1/chat", json=request_data)
    assert response.status_code == 400
    
    # Test with invalid request format
    response = client.post("/v1/chat", json={"invalid": "data"})
    assert response.status_code == 422

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 