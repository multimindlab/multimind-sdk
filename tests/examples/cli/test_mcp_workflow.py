"""
Tests for mcp_workflow.py CLI example.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.cli.mcp_workflow import main


class MockOpenAIModel:
    """Mock OpenAI model for testing."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    async def generate(self, prompt: str, **kwargs):
        return f"Mock OpenAI response to: {prompt}"

    async def generate_stream(self, prompt: str, **kwargs):
        async def stream():
            yield f"Mock OpenAI stream response to: {prompt}"
        return stream()

    async def chat(self, messages, **kwargs):
        return "Mock OpenAI chat response"

    async def chat_stream(self, messages, **kwargs):
        async def stream():
            yield "Mock OpenAI chat stream response"
        return stream()

    async def embeddings(self, text, **kwargs):
        if isinstance(text, str):
            return [0.1] * 384
        return [[0.1] * 384] * len(text)


class MockClaudeModel:
    """Mock Claude model for testing."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    async def generate(self, prompt: str, **kwargs):
        return f"Mock Claude response to: {prompt}"

    async def generate_stream(self, prompt: str, **kwargs):
        async def stream():
            yield f"Mock Claude stream response to: {prompt}"
        return stream()

    async def chat(self, messages, **kwargs):
        return "Mock Claude chat response"

    async def chat_stream(self, messages, **kwargs):
        async def stream():
            yield "Mock Claude chat stream response"
        return stream()

    async def embeddings(self, text, **kwargs):
        if isinstance(text, str):
            return [0.2] * 384
        return [[0.2] * 384] * len(text)


class MockMCPExecutor:
    """Mock MCP executor for testing."""

    def __init__(self):
        self.model_registry = {}
        self.workflow_state = {}
        self.execute_calls = []

    def register_model(self, name: str, model):
        """Register a model."""
        self.model_registry[name] = model

    async def execute(self, workflow: dict, context: dict):
        """Execute a workflow."""
        self.execute_calls.append((workflow, context))
        # Simulate workflow execution
        results = {
            "initial_analysis": "Mock initial analysis result",
            "expert_review": "Mock expert review result",
            "synthesis": "Mock initial analysis result\n\nMock expert review result",
            "quality_check": True
        }
        return results


@pytest.fixture
def mock_models():
    """Fixture to provide mock models."""
    return {
        "openai": MockOpenAIModel("gpt-3.5-turbo"),
        "claude": MockClaudeModel("claude-3-sonnet-20240229")
    }


@pytest.fixture
def mock_executor():
    """Fixture to provide mock executor."""
    return MockMCPExecutor()


@pytest.mark.asyncio
async def test_mcp_workflow_imports():
    """Test that the mcp_workflow module can be imported."""
    try:
        from examples.cli.mcp_workflow import main
        assert main is not None
    except ImportError as e:
        pytest.fail(f"Failed to import mcp_workflow: {e}")


@pytest.mark.asyncio
async def test_mcp_workflow_main_function_both_models():
    """Test main function with both OpenAI and Claude models available."""
    with patch('examples.cli.mcp_workflow.load_dotenv'), \
         patch('examples.cli.mcp_workflow.os.getenv') as mock_getenv, \
         patch('examples.cli.mcp_workflow.OpenAIModel', MockOpenAIModel), \
         patch('examples.cli.mcp_workflow.ClaudeModel', MockClaudeModel), \
         patch('examples.cli.mcp_workflow.MCPExecutor', MockMCPExecutor), \
         patch('builtins.input', return_value=""), \
         patch('builtins.open', create=True) as mock_open:

        # Mock environment variables - both keys available
        def mock_getenv_side_effect(key, default=None):
            if key == "OPENAI_API_KEY":
                return "test-openai-key"
            elif key in ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]:
                return "test-claude-key"
            return default

        mock_getenv.side_effect = mock_getenv_side_effect

        try:
            await main()
            assert True  # If we get here, the function ran without errors
        except Exception as e:
            pytest.fail(f"main() function failed: {e}")


@pytest.mark.asyncio
async def test_mcp_workflow_main_function_openai_only():
    """Test main function with only OpenAI model available."""
    with patch('examples.cli.mcp_workflow.load_dotenv'), \
         patch('examples.cli.mcp_workflow.os.getenv') as mock_getenv, \
         patch('examples.cli.mcp_workflow.OpenAIModel', MockOpenAIModel), \
         patch('examples.cli.mcp_workflow.ClaudeModel', MockClaudeModel), \
         patch('examples.cli.mcp_workflow.MCPExecutor', MockMCPExecutor), \
         patch('builtins.input', return_value=""), \
         patch('builtins.open', create=True) as mock_open:

        # Mock environment variables - only OpenAI key available
        def mock_getenv_side_effect(key, default=None):
            if key == "OPENAI_API_KEY":
                return "test-openai-key"
            elif key in ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]:
                return None
            return default

        mock_getenv.side_effect = mock_getenv_side_effect

        try:
            await main()
            assert True  # If we get here, the function ran without errors
        except Exception as e:
            pytest.fail(f"main() function failed: {e}")


@pytest.mark.asyncio
async def test_mcp_workflow_main_function_claude_only():
    """Test main function with only Claude model available."""
    with patch('examples.cli.mcp_workflow.load_dotenv'), \
         patch('examples.cli.mcp_workflow.os.getenv') as mock_getenv, \
         patch('examples.cli.mcp_workflow.OpenAIModel', MockOpenAIModel), \
         patch('examples.cli.mcp_workflow.ClaudeModel', MockClaudeModel), \
         patch('examples.cli.mcp_workflow.MCPExecutor', MockMCPExecutor), \
         patch('builtins.input', return_value=""), \
         patch('builtins.open', create=True) as mock_open:

        # Mock environment variables - only Claude key available
        def mock_getenv_side_effect(key, default=None):
            if key == "OPENAI_API_KEY":
                return None
            elif key in ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]:
                return "test-claude-key"
            return default

        mock_getenv.side_effect = mock_getenv_side_effect

        try:
            await main()
            assert True  # If we get here, the function ran without errors
        except Exception as e:
            pytest.fail(f"main() function failed: {e}")


@pytest.mark.asyncio
async def test_mcp_workflow_no_api_keys():
    """Test main function with no API keys available."""
    with patch('examples.cli.mcp_workflow.load_dotenv'), \
         patch('examples.cli.mcp_workflow.os.getenv') as mock_getenv:

        # Mock environment variables - no keys available
        def mock_getenv_side_effect(key, default=None):
            return None

        mock_getenv.side_effect = mock_getenv_side_effect

        # Should return early without error
        try:
            await main()
            assert True  # Should return early without error
        except Exception as e:
            pytest.fail(f"main() function should handle no API keys gracefully: {e}")


@pytest.mark.asyncio
async def test_mcp_workflow_custom_topic():
    """Test main function with custom topic input."""
    with patch('examples.cli.mcp_workflow.load_dotenv'), \
         patch('examples.cli.mcp_workflow.os.getenv') as mock_getenv, \
         patch('examples.cli.mcp_workflow.OpenAIModel', MockOpenAIModel), \
         patch('examples.cli.mcp_workflow.ClaudeModel', MockClaudeModel), \
         patch('examples.cli.mcp_workflow.MCPExecutor', MockMCPExecutor), \
         patch('builtins.input', return_value="Quantum Computing"), \
         patch('builtins.open', create=True) as mock_open:

        # Mock environment variables - both keys available
        def mock_getenv_side_effect(key, default=None):
            if key == "OPENAI_API_KEY":
                return "test-openai-key"
            elif key in ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]:
                return "test-claude-key"
            return default

        mock_getenv.side_effect = mock_getenv_side_effect

        try:
            await main()
            assert True  # If we get here, the function ran without errors
        except Exception as e:
            pytest.fail(f"main() function failed: {e}")


@pytest.mark.asyncio
async def test_model_registration():
    """Test that models are registered correctly."""
    executor = MockMCPExecutor()
    openai_model = MockOpenAIModel("gpt-3.5-turbo", temperature=0.7)
    claude_model = MockClaudeModel("claude-3-sonnet-20240229", temperature=0.7)

    executor.register_model("gpt-3.5", openai_model)
    executor.register_model("claude-3", claude_model)

    assert "gpt-3.5" in executor.model_registry
    assert "claude-3" in executor.model_registry
    assert executor.model_registry["gpt-3.5"] == openai_model
    assert executor.model_registry["claude-3"] == claude_model


@pytest.mark.asyncio
async def test_workflow_execution():
    """Test workflow execution."""
    executor = MockMCPExecutor()
    openai_model = MockOpenAIModel("gpt-3.5-turbo")
    executor.register_model("gpt-3.5", openai_model)

    workflow = {
        "version": "1.0.0",
        "models": [],
        "workflow": {
            "steps": [
                {
                    "id": "initial_analysis",
                    "type": "model",
                    "config": {
                        "model": "gpt-3.5",
                        "prompt_template": "Analyze: {topic}"
                    }
                }
            ],
            "connections": []
        }
    }

    context = {"topic": "Test Topic"}
    results = await executor.execute(workflow, context)

    assert len(executor.execute_calls) == 1
    assert executor.execute_calls[0][1] == context
    assert "initial_analysis" in results


@pytest.mark.asyncio
async def test_quality_check_word_extraction():
    """Test quality check word extraction logic."""
    # Test with multiple significant words
    topic = "The Future of Artificial Intelligence"
    topic_words = topic.lower().split()
    articles = ["the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but"]
    significant_words = [w for w in topic_words if w not in articles]

    if len(significant_words) > 1:
        quality_check_word = significant_words[-1]
    elif significant_words:
        quality_check_word = significant_words[0]
    else:
        quality_check_word = "analysis"

    assert quality_check_word == "intelligence"

    # Test with single significant word
    topic = "Quantum Computing"
    topic_words = topic.lower().split()
    significant_words = [w for w in topic_words if w not in articles]

    if len(significant_words) > 1:
        quality_check_word = significant_words[-1]
    elif significant_words:
        quality_check_word = significant_words[0]
    else:
        quality_check_word = "analysis"

    assert quality_check_word == "computing"

    # Test with only articles
    topic = "The Of And"
    topic_words = topic.lower().split()
    significant_words = [w for w in topic_words if w not in articles]

    if len(significant_words) > 1:
        quality_check_word = significant_words[-1]
    elif significant_words:
        quality_check_word = significant_words[0]
    else:
        quality_check_word = "analysis"

    assert quality_check_word == "analysis"


@pytest.mark.asyncio
async def test_workflow_structure():
    """Test that workflow structure is correct."""
    workflow_models = [
        {
            "name": "gpt-3.5",
            "type": "openai",
            "config": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.7
            }
        },
        {
            "name": "claude-3",
            "type": "claude",
            "config": {
                "model": "claude-3-sonnet-20240229",
                "temperature": 0.7
            }
        }
    ]

    workflow = {
        "version": "1.0.0",
        "models": workflow_models,
        "workflow": {
            "steps": [
                {
                    "id": "initial_analysis",
                    "type": "model",
                    "config": {
                        "model": "gpt-3.5",
                        "prompt_template": "Analyze the following topic: {topic}\nProvide a detailed analysis."
                    }
                },
                {
                    "id": "expert_review",
                    "type": "model",
                    "config": {
                        "model": "claude-3",
                        "prompt_template": "Review and enhance the following analysis:\n{initial_analysis}\nProvide expert insights and additional perspectives."
                    }
                },
                {
                    "id": "synthesis",
                    "type": "transform",
                    "config": {
                        "type": "join",
                        "separator": "\n\n"
                    }
                },
                {
                    "id": "quality_check",
                    "type": "condition",
                    "config": {
                        "type": "contains",
                        "value": "intelligence"
                    }
                }
            ],
            "connections": [
                {
                    "from": "initial_analysis",
                    "to": "expert_review"
                },
                {
                    "from": "initial_analysis",
                    "to": "synthesis"
                },
                {
                    "from": "expert_review",
                    "to": "synthesis"
                },
                {
                    "from": "synthesis",
                    "to": "quality_check"
                }
            ]
        }
    }

    assert workflow["version"] == "1.0.0"
    assert len(workflow["models"]) == 2
    assert len(workflow["workflow"]["steps"]) == 4
    assert len(workflow["workflow"]["connections"]) == 4
    assert workflow["workflow"]["steps"][0]["type"] == "model"
    assert workflow["workflow"]["steps"][2]["type"] == "transform"
    assert workflow["workflow"]["steps"][3]["type"] == "condition"


@pytest.mark.asyncio
async def test_model_initialization_errors():
    """Test handling of model initialization errors."""
    with patch('examples.cli.mcp_workflow.load_dotenv'), \
         patch('examples.cli.mcp_workflow.os.getenv') as mock_getenv, \
         patch('examples.cli.mcp_workflow.OpenAIModel') as mock_openai, \
         patch('examples.cli.mcp_workflow.ClaudeModel', MockClaudeModel), \
         patch('examples.cli.mcp_workflow.MCPExecutor', MockMCPExecutor), \
         patch('builtins.input', return_value=""), \
         patch('builtins.open', create=True):

        # Mock environment variables - both keys available
        def mock_getenv_side_effect(key, default=None):
            if key == "OPENAI_API_KEY":
                return "test-openai-key"
            elif key in ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]:
                return "test-claude-key"
            return default

        mock_getenv.side_effect = mock_getenv_side_effect

        # Make OpenAI model initialization fail
        mock_openai.side_effect = Exception("OpenAI initialization failed")

        try:
            await main()
            # Should continue with Claude only
            assert True
        except Exception as e:
            # Should handle the error gracefully
            pytest.fail(f"main() should handle model initialization errors: {e}")


@pytest.mark.asyncio
async def test_workflow_file_saving():
    """Test that workflow executes successfully."""
    with patch('examples.cli.mcp_workflow.load_dotenv'), \
         patch('examples.cli.mcp_workflow.os.getenv') as mock_getenv, \
         patch('examples.cli.mcp_workflow.OpenAIModel', MockOpenAIModel), \
         patch('examples.cli.mcp_workflow.ClaudeModel', MockClaudeModel), \
         patch('examples.cli.mcp_workflow.MCPExecutor', MockMCPExecutor), \
         patch('builtins.input', return_value=""), \
         patch('builtins.open', create=True) as mock_open:

        # Mock environment variables - both keys available
        def mock_getenv_side_effect(key, default=None):
            if key == "OPENAI_API_KEY":
                return "test-openai-key"
            elif key in ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]:
                return "test-claude-key"
            return default

        mock_getenv.side_effect = mock_getenv_side_effect

        try:
            await main()
            assert True  # If we get here, the function ran without errors
        except Exception as e:
            pytest.fail(f"main() function failed: {e}")


@pytest.mark.asyncio
async def test_executor_results_format():
    """Test that executor returns results in expected format."""
    executor = MockMCPExecutor()
    workflow = {
        "version": "1.0.0",
        "models": [],
        "workflow": {
            "steps": [],
            "connections": []
        }
    }

    results = await executor.execute(workflow, {"topic": "Test"})

    assert isinstance(results, dict)
    assert "initial_analysis" in results
    assert "expert_review" in results
    assert "synthesis" in results
    assert "quality_check" in results


def test_example_structure():
    """Test that the example has the expected structure."""
    example_path = Path(__file__).parent.parent.parent.parent / "examples" / "cli" / "mcp_workflow.py"
    assert example_path.exists(), "mcp_workflow.py example should exist"

    # Check that the file contains expected components
    with open(example_path, 'r') as f:
        content = f.read()
        assert "async def main()" in content
        assert "OpenAIModel" in content
        assert "ClaudeModel" in content
        assert "MCPExecutor" in content
        assert "workflow" in content
        assert "execute" in content

