import pytest
from unittest.mock import patch, MagicMock
from model_wrapper import ModelWrapper

@pytest.fixture
def wrapper():
    with patch('model_wrapper.load_dotenv'):
        return ModelWrapper()

def test_available_models_with_no_keys(wrapper):
    """Test available_models with no API keys."""
    wrapper.openai_key = None
    wrapper.claude_key = None
    wrapper.hf_token = None
    
    with patch('subprocess.run', side_effect=FileNotFoundError):
        available = wrapper.available_models()
        assert len(available) == 0

def test_available_models_with_all_keys(wrapper):
    """Test available_models with all API keys."""
    wrapper.openai_key = "test"
    wrapper.claude_key = "test"
    wrapper.hf_token = "test"
    
    with patch('subprocess.run', return_value=MagicMock(returncode=0)):
        available = wrapper.available_models()
        assert "openai" in available
        assert "claude" in available
        assert "huggingface" in available
        assert "ollama" in available

def test_query_model_unavailable(wrapper):
    """Test querying an unavailable model."""
    wrapper.openai_key = None
    result = wrapper.query_model("openai", "test prompt")
    assert result["status"] == "error"
    assert "not available" in result["error"]

@patch('openai.ChatCompletion.create')
def test_query_openai(mock_create, wrapper):
    """Test OpenAI query."""
    mock_create.return_value = MagicMock(
        choices=[MagicMock(message={'content': 'test response'})]
    )
    wrapper.openai_key = "test"
    
    result = wrapper.query_model("openai", "test prompt")
    assert result["status"] == "success"
    assert result["response"] == "test response"
    mock_create.assert_called_once()

@patch('subprocess.run')
def test_query_ollama(mock_run, wrapper):
    """Test Ollama query."""
    mock_run.return_value = MagicMock(stdout="test response", returncode=0)
    
    with patch('subprocess.run', return_value=MagicMock(returncode=0)):
        result = wrapper.query_model("ollama", "test prompt")
        assert result["status"] == "success"
        assert result["response"] == "test response"

def test_query_model_error_handling(wrapper):
    """Test error handling in query_model."""
    wrapper.openai_key = "test"
    with patch('model_wrapper.ModelWrapper.query_openai', side_effect=Exception("test error")):
        result = wrapper.query_model("openai", "test prompt")
        assert result["status"] == "error"
        assert result["error"] == "test error"