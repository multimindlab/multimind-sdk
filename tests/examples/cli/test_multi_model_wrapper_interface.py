"""
Tests for multi_model_wrapper_interface.py CLI example.
"""

import os
import subprocess
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add examples directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


class MockCompletions:
    """Mock completions class for testing."""
    def create(self, model, messages):
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = f"Mock ChatGPT response to: {messages[0]['content']}"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response


class MockChatCompletions:
    """Mock chat completions class for testing."""
    def __init__(self):
        self.completions = MockCompletions()


class MockOpenAIClient:
    """Mock OpenAI client for testing."""

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.chat = MockChatCompletions()


@pytest.fixture
def mock_openai_client():
    """Fixture to provide a mock OpenAI client."""
    return MockOpenAIClient()


@pytest.fixture
def mock_env_with_api_key(monkeypatch):
    """Fixture to set OPENAI_API_KEY environment variable."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-123")
    yield
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def mock_env_without_api_key(monkeypatch):
    """Fixture to ensure OPENAI_API_KEY is not set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield


def test_imports():
    """Test that the module can be imported."""
    try:
        from examples.cli.multi_model_wrapper_interface import main, query_chatgpt, query_ollama
        assert query_chatgpt is not None
        assert query_ollama is not None
        assert main is not None
    except ImportError as e:
        pytest.fail(f"Failed to import multi_model_wrapper_interface: {e}")


def test_query_chatgpt_with_api_key(mock_env_with_api_key):
    """Test query_chatgpt function with valid API key."""
    from examples.cli.multi_model_wrapper_interface import query_chatgpt

    with patch('examples.cli.multi_model_wrapper_interface.OpenAI') as MockOpenAI:
        mock_client = MockOpenAIClient()
        MockOpenAI.return_value = mock_client

        response = query_chatgpt("Hello, how are you?")

        assert response is not None
        assert "Mock ChatGPT response" in response
        assert "Hello, how are you?" in response
        MockOpenAI.assert_called_once_with(api_key="test-api-key-123")


def test_query_chatgpt_without_api_key(mock_env_without_api_key):
    """Test query_chatgpt function without API key raises error."""
    from examples.cli.multi_model_wrapper_interface import query_chatgpt

    with patch('examples.cli.multi_model_wrapper_interface.load_dotenv'):
        with pytest.raises(ValueError) as exc_info:
            query_chatgpt("Hello")

        assert "OPENAI_API_KEY is not set" in str(exc_info.value)


def test_query_chatgpt_loads_dotenv(mock_env_with_api_key):
    """Test that query_chatgpt loads .env file."""
    from examples.cli.multi_model_wrapper_interface import query_chatgpt

    with patch('examples.cli.multi_model_wrapper_interface.load_dotenv') as mock_load_dotenv, \
         patch('examples.cli.multi_model_wrapper_interface.OpenAI') as MockOpenAI:
        mock_client = MockOpenAIClient()
        MockOpenAI.return_value = mock_client

        query_chatgpt("Test prompt")

        mock_load_dotenv.assert_called_once()


def test_query_chatgpt_calls_openai_correctly(mock_env_with_api_key):
    """Test that query_chatgpt calls OpenAI API with correct parameters."""
    from examples.cli.multi_model_wrapper_interface import query_chatgpt

    with patch('examples.cli.multi_model_wrapper_interface.OpenAI') as MockOpenAI:
        # Create a mock client with a mock create method
        mock_create = Mock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="Mock response"))]
        ))
        mock_completions = Mock()
        mock_completions.create = mock_create
        mock_chat = Mock()
        mock_chat.completions = mock_completions
        mock_client = Mock()
        mock_client.chat = mock_chat
        MockOpenAI.return_value = mock_client

        prompt = "What is Python?"
        response = query_chatgpt(prompt)

        # Verify OpenAI client was created with API key
        MockOpenAI.assert_called_once_with(api_key="test-api-key-123")

        # Verify chat.completions.create was called with correct parameters
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs['model'] == "gpt-4"
        assert call_kwargs['messages'][0]['content'] == prompt
        assert call_kwargs['messages'][0]['role'] == "user"
        assert response == "Mock response"


def test_query_ollama_success():
    """Test query_ollama function with successful subprocess."""
    from examples.cli.multi_model_wrapper_interface import query_ollama

    mock_result = Mock()
    mock_result.stdout = "Mock Ollama response\n"
    mock_result.returncode = 0

    with patch('examples.cli.multi_model_wrapper_interface.subprocess.run') as mock_run:
        mock_run.return_value = mock_result

        response = query_ollama("Hello")

        assert response == "Mock Ollama response\n"
        mock_run.assert_called_once_with(
            ["ollama", "run", "mistral", "Hello"],
            capture_output=True,
            text=True
        )


def test_query_ollama_with_different_prompt():
    """Test query_ollama with different prompts."""
    from examples.cli.multi_model_wrapper_interface import query_ollama

    mock_result = Mock()
    mock_result.stdout = "Python is a programming language\n"
    mock_result.returncode = 0

    with patch('examples.cli.multi_model_wrapper_interface.subprocess.run') as mock_run:
        mock_run.return_value = mock_result

        response = query_ollama("What is Python?")

        assert "Python is a programming language" in response
        mock_run.assert_called_once_with(
            ["ollama", "run", "mistral", "What is Python?"],
            capture_output=True,
            text=True
        )


def test_main_with_chatgpt_model(capsys, mock_env_with_api_key):
    """Test main function with chatgpt model."""
    from examples.cli.multi_model_wrapper_interface import main

    with patch('examples.cli.multi_model_wrapper_interface.OpenAI') as MockOpenAI, \
         patch('sys.argv', ['multi_model_wrapper_interface.py', '--model', 'chatgpt', '--prompt', 'Hello']):
        mock_client = MockOpenAIClient()
        MockOpenAI.return_value = mock_client

        main()

        captured = capsys.readouterr()
        assert "--- Response ---" in captured.out
        assert "Mock ChatGPT response" in captured.out


def test_main_with_ollama_model(capsys):
    """Test main function with ollama model."""
    from examples.cli.multi_model_wrapper_interface import main

    mock_result = Mock()
    mock_result.stdout = "Mock Ollama response\n"
    mock_result.returncode = 0

    with patch('examples.cli.multi_model_wrapper_interface.subprocess.run') as mock_run, \
         patch('sys.argv', ['multi_model_wrapper_interface.py', '--model', 'ollama', '--prompt', 'Hello']):
        mock_run.return_value = mock_result

        main()

        captured = capsys.readouterr()
        assert "--- Response ---" in captured.out
        assert "Mock Ollama response" in captured.out


def test_main_missing_required_arguments(capsys):
    """Test main function with missing required arguments."""
    from examples.cli.multi_model_wrapper_interface import main

    with patch('sys.argv', ['multi_model_wrapper_interface.py']):
        with pytest.raises(SystemExit):
            main()


def test_main_missing_model_argument(capsys):
    """Test main function with missing --model argument."""
    from examples.cli.multi_model_wrapper_interface import main

    with patch('sys.argv', ['multi_model_wrapper_interface.py', '--prompt', 'Hello']):
        with pytest.raises(SystemExit):
            main()


def test_main_missing_prompt_argument(capsys):
    """Test main function with missing --prompt argument."""
    from examples.cli.multi_model_wrapper_interface import main

    with patch('sys.argv', ['multi_model_wrapper_interface.py', '--model', 'chatgpt']):
        with pytest.raises(SystemExit):
            main()


def test_main_invalid_model_argument(capsys):
    """Test main function with invalid model argument."""
    from examples.cli.multi_model_wrapper_interface import main

    with patch('sys.argv', ['multi_model_wrapper_interface.py', '--model', 'invalid', '--prompt', 'Hello']):
        with pytest.raises(SystemExit):
            main()


def test_main_help_message(capsys):
    """Test that help message is displayed correctly."""
    from examples.cli.multi_model_wrapper_interface import main

    with patch('sys.argv', ['multi_model_wrapper_interface.py', '--help']):
        with pytest.raises(SystemExit):
            main()

        captured = capsys.readouterr()
        assert "Query ChatGPT and Mistral using CLI" in captured.out
        assert "--model" in captured.out
        assert "--prompt" in captured.out
        assert "chatgpt" in captured.out or "ollama" in captured.out


def test_query_chatgpt_error_handling(mock_env_with_api_key):
    """Test query_chatgpt error handling when API call fails."""
    from examples.cli.multi_model_wrapper_interface import query_chatgpt

    with patch('examples.cli.multi_model_wrapper_interface.OpenAI') as MockOpenAI:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error: Rate limit exceeded")
        MockOpenAI.return_value = mock_client

        with pytest.raises(Exception) as exc_info:
            query_chatgpt("Hello")

        assert "API Error" in str(exc_info.value)


def test_query_ollama_subprocess_error():
    """Test query_ollama when subprocess fails."""
    from examples.cli.multi_model_wrapper_interface import query_ollama

    mock_result = Mock()
    mock_result.stdout = ""
    mock_result.returncode = 1
    mock_result.stderr = "Error: command not found"

    with patch('examples.cli.multi_model_wrapper_interface.subprocess.run') as mock_run:
        mock_run.return_value = mock_result

        # The function should still return the stdout (which is empty in this case)
        response = query_ollama("Hello")

        assert response == ""
        mock_run.assert_called_once()


def test_file_structure():
    """Test that the CLI file has the expected structure."""
    cli_path = Path(__file__).parent.parent.parent.parent / "examples" / "cli" / "multi_model_wrapper_interface.py"
    assert cli_path.exists(), "multi_model_wrapper_interface.py should exist"

    # Check that the file contains expected components
    with open(cli_path, 'r') as f:
        content = f.read()
        assert "def query_chatgpt" in content
        assert "def query_ollama" in content
        assert "def main" in content
        assert "argparse" in content
        assert "OpenAI" in content
        assert "load_dotenv" in content
        assert "--model" in content
        assert "--prompt" in content


def test_query_chatgpt_with_empty_api_key(monkeypatch):
    """Test query_chatgpt with empty API key."""
    from examples.cli.multi_model_wrapper_interface import query_chatgpt

    monkeypatch.setenv("OPENAI_API_KEY", "")

    with patch('examples.cli.multi_model_wrapper_interface.load_dotenv'):
        with pytest.raises(ValueError) as exc_info:
            query_chatgpt("Hello")

        assert "OPENAI_API_KEY is not set" in str(exc_info.value)


def test_query_chatgpt_api_key_from_env_after_dotenv(monkeypatch):
    """Test that API key is loaded from environment after dotenv."""
    from examples.cli.multi_model_wrapper_interface import query_chatgpt

    # Set env var
    monkeypatch.setenv("OPENAI_API_KEY", "env-api-key")

    def mock_load_dotenv():
        # Simulate dotenv setting a different key
        monkeypatch.setenv("OPENAI_API_KEY", "dotenv-api-key")

    with patch('examples.cli.multi_model_wrapper_interface.load_dotenv', side_effect=mock_load_dotenv), \
         patch('examples.cli.multi_model_wrapper_interface.OpenAI') as MockOpenAI:
        mock_client = MockOpenAIClient()
        MockOpenAI.return_value = mock_client

        query_chatgpt("Test")

        # Should use the key from environment (after dotenv potentially modifies it)
        MockOpenAI.assert_called()

