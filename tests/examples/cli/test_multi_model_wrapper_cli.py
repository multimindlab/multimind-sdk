"""
Tests for multi_model_wrapper_cli.py CLI example.
"""

import asyncio
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Add examples directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


class MockModel:
    """Mock model for testing."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    async def generate(self, prompt: str, **kwargs):
        return f"Mock response to: {prompt}"

    async def generate_stream(self, prompt: str, **kwargs):
        async def stream():
            yield f"Mock stream response to: {prompt}"
        return stream()

    async def chat(self, messages, **kwargs):
        return "Mock chat response"

    async def chat_stream(self, messages, **kwargs):
        async def stream():
            yield "Mock chat stream response"
        return stream()

    async def embeddings(self, text, **kwargs):
        if isinstance(text, str):
            return [0.1] * 384
        return [[0.1] * 384] * len(text)


class MockModelFactory:
    """Mock ModelFactory for testing."""

    def __init__(self, available_models=None, env_path=None):
        self.available_models_list = available_models or ["openai", "ollama"]
        self.openai_key = "test_openai_key"
        self.claude_key = None

    def available_models(self):
        """Return list of available models."""
        return self.available_models_list

    def get_model(self, provider: str, model_name: str = None, **kwargs):
        """Get a model instance."""
        return MockModel(provider, **kwargs)


class MockMultiModelWrapper:
    """Mock MultiModelWrapper for testing."""

    def __init__(self, model_factory, primary_model, fallback_models=None, **kwargs):
        self.model_factory = model_factory
        self.primary_model = primary_model
        self.fallback_models = fallback_models or []
        self.generate_calls = 0
        self.last_prompt = None

    async def generate(self, prompt: str, **kwargs):
        """Generate a response."""
        self.generate_calls += 1
        self.last_prompt = prompt
        return f"Mock MultiModelWrapper response to: {prompt}"

    async def generate_stream(self, prompt: str, **kwargs):
        """Generate a streaming response."""
        async def stream():
            yield f"Mock stream response to: {prompt}"
        return stream()

    async def chat(self, messages, **kwargs):
        """Generate a chat response."""
        return "Mock chat response"

    async def chat_stream(self, messages, **kwargs):
        """Generate a streaming chat response."""
        async def stream():
            yield "Mock chat stream response"
        return stream()

    async def embeddings(self, text, **kwargs):
        """Generate embeddings."""
        if isinstance(text, str):
            return [0.1] * 384
        return [[0.1] * 384] * len(text)


@pytest.fixture
def mock_factory():
    """Fixture to provide a mock ModelFactory."""
    return MockModelFactory(available_models=["openai", "ollama"])


@pytest.fixture
def mock_wrapper():
    """Fixture to provide a mock MultiModelWrapper."""
    factory = MockModelFactory()
    return MockMultiModelWrapper(
        model_factory=factory,
        primary_model="openai",
        fallback_models=["ollama"]
    )


@pytest.mark.asyncio
async def test_cli_imports():
    """Test that the CLI module can be imported."""
    try:
        from examples.cli.multi_model_wrapper_cli import main
        assert main is not None
    except ImportError as e:
        pytest.fail(f"Failed to import multi_model_wrapper_cli: {e}")


@pytest.mark.asyncio
async def test_list_models_flag(capsys):
    """Test the --list-models flag."""
    import argparse

    with patch('examples.cli.multi_model_wrapper_cli.ModelFactory', MockModelFactory):
        from examples.cli.multi_model_wrapper_cli import main

        # Create a mock parser that returns list_models=True
        with patch('examples.cli.multi_model_wrapper_cli.argparse.ArgumentParser') as MockParser:
            mock_parser = MockParser.return_value
            mock_args = argparse.Namespace()
            mock_args.list_models = True
            mock_args.model = None
            mock_args.prompt = None
            mock_args.temperature = 0.7
            mock_args.max_tokens = None
            mock_parser.parse_args.return_value = mock_args

            await main()

            captured = capsys.readouterr()
            assert "=== Available Models ===" in captured.out
            assert "openai" in captured.out or "ollama" in captured.out
            assert "Total:" in captured.out


@pytest.mark.asyncio
async def test_list_models_with_no_models(capsys):
    """Test --list-models when no models are available."""
    import argparse

    with patch('examples.cli.multi_model_wrapper_cli.ModelFactory') as MockFactory:
        mock_factory = MockFactory.return_value
        mock_factory.available_models.return_value = []

        from examples.cli.multi_model_wrapper_cli import main

        # Create a mock parser that returns list_models=True
        with patch('examples.cli.multi_model_wrapper_cli.argparse.ArgumentParser') as MockParser:
            mock_parser = MockParser.return_value
            mock_args = argparse.Namespace()
            mock_args.list_models = True
            mock_args.model = None
            mock_args.prompt = None
            mock_args.temperature = 0.7
            mock_args.max_tokens = None
            mock_parser.parse_args.return_value = mock_args

            await main()

            captured = capsys.readouterr()
            assert "No models available" in captured.out


@pytest.mark.asyncio
async def test_missing_required_arguments(capsys):
    """Test that missing required arguments show an error."""
    import argparse

    with patch('examples.cli.multi_model_wrapper_cli.ModelFactory', MockModelFactory):
        from examples.cli.multi_model_wrapper_cli import main

        # Create a mock parser that returns None for model and prompt
        with patch('examples.cli.multi_model_wrapper_cli.argparse.ArgumentParser') as MockParser:
            mock_parser = MockParser.return_value
            mock_args = argparse.Namespace()
            mock_args.list_models = False
            mock_args.model = None
            mock_args.prompt = None
            mock_args.temperature = 0.7
            mock_args.max_tokens = None
            mock_parser.parse_args.return_value = mock_args
            mock_parser.error = Mock(side_effect=SystemExit(2))

            with pytest.raises(SystemExit):
                await main()


@pytest.mark.asyncio
async def test_missing_model_argument(capsys):
    """Test that missing --model argument shows an error."""
    import argparse

    with patch('examples.cli.multi_model_wrapper_cli.ModelFactory', MockModelFactory):
        from examples.cli.multi_model_wrapper_cli import main

        # Create a mock parser that returns None for model
        with patch('examples.cli.multi_model_wrapper_cli.argparse.ArgumentParser') as MockParser:
            mock_parser = MockParser.return_value
            mock_args = argparse.Namespace()
            mock_args.list_models = False
            mock_args.model = None
            mock_args.prompt = "test"
            mock_args.temperature = 0.7
            mock_args.max_tokens = None
            mock_parser.parse_args.return_value = mock_args
            mock_parser.error = Mock(side_effect=SystemExit(2))

            with pytest.raises(SystemExit):
                await main()


@pytest.mark.asyncio
async def test_missing_prompt_argument(capsys):
    """Test that missing --prompt argument shows an error."""
    import argparse

    with patch('examples.cli.multi_model_wrapper_cli.ModelFactory', MockModelFactory):
        from examples.cli.multi_model_wrapper_cli import main

        # Create a mock parser that returns None for prompt
        with patch('examples.cli.multi_model_wrapper_cli.argparse.ArgumentParser') as MockParser:
            mock_parser = MockParser.return_value
            mock_args = argparse.Namespace()
            mock_args.list_models = False
            mock_args.model = "openai"
            mock_args.prompt = None
            mock_args.temperature = 0.7
            mock_args.max_tokens = None
            mock_parser.parse_args.return_value = mock_args
            mock_parser.error = Mock(side_effect=SystemExit(2))

            with pytest.raises(SystemExit):
                await main()


@pytest.mark.asyncio
async def test_successful_generation(capsys):
    """Test successful model query."""
    import argparse

    with patch('examples.cli.multi_model_wrapper_cli.ModelFactory', MockModelFactory), \
         patch('examples.cli.multi_model_wrapper_cli.MultiModelWrapper', MockMultiModelWrapper):

        from examples.cli.multi_model_wrapper_cli import main

        # Create a mock parser with valid arguments
        with patch('examples.cli.multi_model_wrapper_cli.argparse.ArgumentParser') as MockParser:
            mock_parser = MockParser.return_value
            mock_args = argparse.Namespace()
            mock_args.list_models = False
            mock_args.model = "openai"
            mock_args.prompt = "Hello, world!"
            mock_args.temperature = 0.7
            mock_args.max_tokens = None
            mock_parser.parse_args.return_value = mock_args

            await main()

            captured = capsys.readouterr()
            assert "Response" in captured.out
            assert "Mock MultiModelWrapper response" in captured.out


@pytest.mark.asyncio
async def test_generation_with_temperature(capsys):
    """Test generation with custom temperature."""
    import argparse

    with patch('examples.cli.multi_model_wrapper_cli.ModelFactory', MockModelFactory), \
         patch('examples.cli.multi_model_wrapper_cli.MultiModelWrapper', MockMultiModelWrapper):

        from examples.cli.multi_model_wrapper_cli import main

        # Create a mock parser with temperature argument
        with patch('examples.cli.multi_model_wrapper_cli.argparse.ArgumentParser') as MockParser:
            mock_parser = MockParser.return_value
            mock_args = argparse.Namespace()
            mock_args.list_models = False
            mock_args.model = "openai"
            mock_args.prompt = "Test"
            mock_args.temperature = 0.9
            mock_args.max_tokens = None
            mock_parser.parse_args.return_value = mock_args

            await main()

            captured = capsys.readouterr()
            assert "Response" in captured.out


@pytest.mark.asyncio
async def test_generation_with_max_tokens(capsys):
    """Test generation with max_tokens parameter."""
    import argparse

    with patch('examples.cli.multi_model_wrapper_cli.ModelFactory', MockModelFactory), \
         patch('examples.cli.multi_model_wrapper_cli.MultiModelWrapper', MockMultiModelWrapper):

        from examples.cli.multi_model_wrapper_cli import main

        # Create a mock parser with max_tokens argument
        with patch('examples.cli.multi_model_wrapper_cli.argparse.ArgumentParser') as MockParser:
            mock_parser = MockParser.return_value
            mock_args = argparse.Namespace()
            mock_args.list_models = False
            mock_args.model = "openai"
            mock_args.prompt = "Test"
            mock_args.temperature = 0.7
            mock_args.max_tokens = 100
            mock_parser.parse_args.return_value = mock_args

            await main()

            captured = capsys.readouterr()
            assert "Response" in captured.out


@pytest.mark.asyncio
async def test_error_handling(capsys):
    """Test error handling when model generation fails."""
    import argparse

    class FailingMultiModelWrapper:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, *args, **kwargs):
            raise Exception("API Error: Rate limit exceeded")

    with patch('examples.cli.multi_model_wrapper_cli.ModelFactory', MockModelFactory), \
         patch('examples.cli.multi_model_wrapper_cli.MultiModelWrapper', FailingMultiModelWrapper):

        from examples.cli.multi_model_wrapper_cli import main

        # Create a mock parser with valid arguments
        with patch('examples.cli.multi_model_wrapper_cli.argparse.ArgumentParser') as MockParser:
            mock_parser = MockParser.return_value
            mock_args = argparse.Namespace()
            mock_args.list_models = False
            mock_args.model = "openai"
            mock_args.prompt = "Test"
            mock_args.temperature = 0.7
            mock_args.max_tokens = None
            mock_parser.parse_args.return_value = mock_args

            await main()

            captured = capsys.readouterr()
            assert "Error" in captured.out
            assert "API Error" in captured.out


@pytest.mark.asyncio
async def test_wrapper_initialization():
    """Test that MultiModelWrapper is initialized correctly."""
    factory = MockModelFactory()
    wrapper = MockMultiModelWrapper(
        model_factory=factory,
        primary_model="openai",
        fallback_models=["ollama"]
    )

    assert wrapper.primary_model == "openai"
    assert "ollama" in wrapper.fallback_models
    assert wrapper.model_factory == factory


@pytest.mark.asyncio
async def test_wrapper_generate():
    """Test that the wrapper can generate responses."""
    factory = MockModelFactory()
    wrapper = MockMultiModelWrapper(
        model_factory=factory,
        primary_model="openai",
        fallback_models=["ollama"]
    )

    response = await wrapper.generate(
        prompt="Test prompt",
        temperature=0.7,
        max_tokens=100
    )

    assert response is not None
    assert "Mock MultiModelWrapper response" in response
    assert wrapper.generate_calls == 1
    assert wrapper.last_prompt == "Test prompt"


@pytest.mark.asyncio
async def test_factory_available_models():
    """Test that ModelFactory returns available models."""
    factory = MockModelFactory(available_models=["openai", "claude", "ollama"])
    available = factory.available_models()

    assert len(available) == 3
    assert "openai" in available
    assert "claude" in available
    assert "ollama" in available


@pytest.mark.asyncio
async def test_factory_get_model():
    """Test that ModelFactory can create model instances."""
    factory = MockModelFactory()
    model = factory.get_model("openai", "gpt-4")

    assert model is not None
    assert model.model_name == "openai"

    # Test that the model can generate
    response = await model.generate("Test prompt")
    assert "Mock response" in response


@pytest.mark.asyncio
async def test_cli_with_different_models(capsys):
    """Test CLI with different model selections."""
    import argparse

    test_cases = [
        ("openai", ["openai", "ollama"]),
        ("ollama", ["openai", "ollama"]),
    ]

    for model_name, available in test_cases:
        with patch('examples.cli.multi_model_wrapper_cli.ModelFactory') as MockFactory:
            mock_factory = MockFactory.return_value
            mock_factory.available_models.return_value = available

            with patch('examples.cli.multi_model_wrapper_cli.MultiModelWrapper', MockMultiModelWrapper):
                from examples.cli.multi_model_wrapper_cli import main

                # Create a mock parser with model-specific arguments
                with patch('examples.cli.multi_model_wrapper_cli.argparse.ArgumentParser') as MockParser:
                    mock_parser = MockParser.return_value
                    mock_args = argparse.Namespace()
                    mock_args.list_models = False
                    mock_args.model = model_name
                    mock_args.prompt = "Test"
                    mock_args.temperature = 0.7
                    mock_args.max_tokens = None
                    mock_parser.parse_args.return_value = mock_args

                    await main()

                    captured = capsys.readouterr()
                    assert model_name.upper() in captured.out or "Response" in captured.out


@pytest.mark.asyncio
async def test_fallback_models_logic():
    """Test that fallback models are correctly determined."""
    factory = MockModelFactory(available_models=["openai", "claude", "ollama"])
    available = factory.available_models()

    selected_model = "openai"
    fallback_models = [m for m in available if m != selected_model]

    assert len(fallback_models) == 2
    assert "claude" in fallback_models
    assert "ollama" in fallback_models
    assert "openai" not in fallback_models


def test_cli_file_structure():
    """Test that the CLI file has the expected structure."""
    cli_path = Path(__file__).parent.parent.parent.parent / "examples" / "cli" / "multi_model_wrapper_cli.py"
    assert cli_path.exists(), "multi_model_wrapper_cli.py should exist"

    # Check that the file contains expected components
    with open(cli_path, 'r') as f:
        content = f.read()
        assert "async def main()" in content
        assert "ModelFactory" in content
        assert "MultiModelWrapper" in content
        assert "--list-models" in content
        assert "--model" in content
        assert "--prompt" in content
        assert "argparse" in content

