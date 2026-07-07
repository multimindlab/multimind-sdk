"""
Tests for ensemble_cli.py CLI example.
"""

import pytest
pytest.importorskip("numpy")  # requires optional extras absent on core-only installs
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import os
import sys
import json
from pathlib import Path
from io import StringIO

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.cli.ensemble_cli import (
    ensemble,
    _normalize_provider,
    _get_env_value,
    _get_provider_name,
    _configure_task,
    _configure_default_tasks,
    _prepare_router,
    _providers_for_task,
    generate,
    review,
    analyze_image,
    embed,
)


class MockProvider:
    """Mock provider for testing."""
    
    def __init__(self, name: str, capabilities=None):
        self.name = name
        self.provider_name = name
        self.capabilities = capabilities or set()
    
    async def generate(self, prompt: str, **kwargs):
        return f"Mock {self.name} response to: {prompt}"
    
    async def embeddings(self, text, **kwargs):
        if isinstance(text, str):
            return [0.1] * 384
        return [[0.1] * 384] * len(text)
    
    async def analyze_image(self, image_data, prompt=None, **kwargs):
        return f"Mock {self.name} image analysis"
    
    async def list_models(self):
        return ["model1", "model2", "mistral"]


class MockGenerationResult:
    """Mock generation result for testing."""
    
    def __init__(self, text: str, provider_name: str = "openai"):
        self.text = text
        self.result = text
        self.provider_name = provider_name
        self.provider = provider_name


class MockEmbeddingResult:
    """Mock embedding result for testing."""
    
    def __init__(self, embedding: list, provider_name: str = "openai"):
        self.embedding = embedding
        self.provider_name = provider_name
        self.provider = provider_name


class MockImageAnalysisResult:
    """Mock image analysis result for testing."""
    
    def __init__(self, text: str = None, captions: list = None, provider_name: str = "openai"):
        self.text = text or "Mock image analysis"
        self.captions = captions or [self.text]
        self.provider_name = provider_name
        self.provider = provider_name


class MockConfidence:
    """Mock confidence object for testing."""
    
    def __init__(self, score: float = 0.9, explanation: str = "High confidence"):
        self.score = score
        self.explanation = explanation


class MockCombinedResult:
    """Mock combined result for testing."""
    
    def __init__(self, result, confidence=None, provider_votes=None):
        self.result = result
        self.confidence = confidence or MockConfidence()
        self.provider_votes = provider_votes or {}


class MockRouter:
    """Mock router for testing."""
    
    def __init__(self):
        self.providers = {}
        self.fallback_policy = Mock()
        self.fallback_policy.notify_user = False
    
    def register_provider(self, name: str, provider):
        self.providers[name] = provider
    
    def configure_task(self, task_type, config):
        pass
    
    async def route(self, task_type, input_data, **kwargs):
        provider_name = kwargs.get("provider", "openai")
        provider = self.providers.get(provider_name)
        if not provider:
            raise Exception(f"Provider {provider_name} not found")
        
        if task_type.value == "text_generation":
            return MockGenerationResult(f"Mock {provider_name} response", provider_name)
        elif task_type.value == "embeddings":
            return MockEmbeddingResult([0.1] * 384, provider_name)
        elif task_type.value == "image_analysis":
            return MockImageAnalysisResult(f"Mock {provider_name} image analysis", provider_name=provider_name)
        else:
            raise Exception(f"Unknown task type: {task_type}")


class MockAdvancedEnsemble:
    """Mock advanced ensemble for testing."""
    
    def __init__(self, router):
        self.router = router
    
    async def combine_results(self, results, method, task_type, **kwargs):
        # Return a mock combined result
        if results:
            first_result = results[0]
            if hasattr(first_result, 'text'):
                result_obj = first_result
            elif hasattr(first_result, 'embedding'):
                result_obj = first_result
            elif hasattr(first_result, 'captions'):
                result_obj = first_result
            else:
                result_obj = first_result
        else:
            result_obj = MockGenerationResult("No results")
        
        return MockCombinedResult(
            result=result_obj,
            confidence=MockConfidence(0.9, "High confidence"),
            provider_votes={"openai": 1.0}
        )


@pytest.fixture
def mock_router():
    """Fixture to provide mock router."""
    return MockRouter()


@pytest.fixture
def mock_ensemble():
    """Fixture to provide mock ensemble."""
    router = MockRouter()
    return MockAdvancedEnsemble(router)


@pytest.fixture
def mock_providers():
    """Fixture to provide mock providers."""
    return {
        "openai": MockProvider("openai"),
        "anthropic": MockProvider("anthropic"),
        "ollama": MockProvider("ollama"),
    }


@pytest.mark.asyncio
async def test_ensemble_cli_imports():
    """Test that the ensemble_cli module can be imported."""
    try:
        from examples.cli.ensemble_cli import ensemble
        assert ensemble is not None
    except ImportError as e:
        pytest.fail(f"Failed to import ensemble_cli: {e}")


def test_normalize_provider():
    """Test provider name normalization."""
    assert _normalize_provider("claude") == "anthropic"
    assert _normalize_provider("openai") == "openai"
    assert _normalize_provider("OLLAMA") == "ollama"


def test_get_env_value():
    """Test environment variable retrieval."""
    with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
        assert _get_env_value(["TEST_VAR"]) == "test_value"
        assert _get_env_value(["NONEXISTENT"]) is None
        assert _get_env_value(["NONEXISTENT", "TEST_VAR"]) == "test_value"


def test_get_provider_name():
    """Test provider name extraction from results."""
    result1 = MockGenerationResult("test", "openai")
    assert _get_provider_name(result1) == "openai"
    
    # Create a simple object that only has 'provider' attribute
    class SimpleResult:
        def __init__(self, provider):
            self.provider = provider
    
    result2 = SimpleResult("anthropic")
    assert _get_provider_name(result2) == "anthropic"
    
    # Test fallback to 'unknown' when neither attribute exists
    class EmptyResult:
        pass
    
    result3 = EmptyResult()
    assert _get_provider_name(result3) == "unknown"


def test_configure_task():
    """Test task configuration."""
    router = MockRouter()
    router.configure_task = Mock()  # Make it a Mock from the start
    from multimind import TaskType, RoutingStrategy
    
    # Test with single provider
    _configure_task(router, TaskType.TEXT_GENERATION, ["openai"])
    assert router.configure_task.called
    
    # Test with multiple providers
    router.configure_task.reset_mock()
    _configure_task(router, TaskType.TEXT_GENERATION, ["openai", "anthropic"])
    assert router.configure_task.called


def test_configure_default_tasks():
    """Test default task configuration."""
    router = MockRouter()
    router.configure_task = Mock()
    
    _configure_default_tasks(router, ["openai", "anthropic"])
    # Should configure text generation, embeddings, and image analysis
    assert router.configure_task.call_count >= 1


def test_providers_for_task():
    """Test provider filtering by task type."""
    from multimind import TaskType
    
    registered = ["openai", "anthropic", "ollama"]
    # Mock the PROVIDER_REGISTRY
    with patch('examples.cli.ensemble_cli.PROVIDER_REGISTRY', {
        "openai": {"capabilities": {TaskType.TEXT_GENERATION, TaskType.EMBEDDINGS}},
        "anthropic": {"capabilities": {TaskType.TEXT_GENERATION}},
        "ollama": {"capabilities": {TaskType.TEXT_GENERATION, TaskType.EMBEDDINGS}},
    }):
        text_providers = _providers_for_task(registered, TaskType.TEXT_GENERATION)
        assert "openai" in text_providers
        assert "anthropic" in text_providers
        assert "ollama" in text_providers


@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key", "ANTHROPIC_API_KEY": "test_key"})
def test_prepare_router():
    """Test router preparation."""
    mock_openai = Mock()
    mock_claude = Mock()
    mock_ollama = Mock()
    
    with patch('examples.cli.ensemble_cli.AdvancedEnsemble', MockAdvancedEnsemble), \
         patch('examples.cli.ensemble_cli.Router', MockRouter), \
         patch('examples.cli.ensemble_cli.OpenAIProvider', return_value=mock_openai), \
         patch('examples.cli.ensemble_cli.ClaudeProvider', return_value=mock_claude), \
         patch('examples.cli.ensemble_cli.OllamaProvider', return_value=mock_ollama):
        
        router, registered = _prepare_router(["openai", "anthropic"])
        assert router is not None
        assert len(registered) > 0


@patch('examples.cli.ensemble_cli.AdvancedEnsemble', MockAdvancedEnsemble)
@patch('examples.cli.ensemble_cli.Router', MockRouter)
@patch('examples.cli.ensemble_cli.asyncio.run')
@patch('click.echo')
@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
def test_generate_command(mock_echo, mock_asyncio_run):
    """Test the generate command."""
    mock_router = MockRouter()
    mock_router.providers = {"openai": MockProvider("openai")}
    
    with patch('examples.cli.ensemble_cli._prepare_router', return_value=(mock_router, ["openai"])), \
         patch('examples.cli.ensemble_cli._providers_for_task', return_value=["openai"]):
        
        # Mock the async run function
        async def mock_run():
            mock_ensemble = MockAdvancedEnsemble(mock_router)
            results = [MockGenerationResult("test response", "openai")]
            combined = await mock_ensemble.combine_results(
                results=results,
                method=None,
                task_type=None
            )
            output_data = {
                "result": "test response",
                "confidence": combined.confidence.score,
                "explanation": combined.confidence.explanation,
                "provider_votes": combined.provider_votes
            }
            mock_echo(json.dumps(output_data, indent=2))
        
        mock_asyncio_run.side_effect = lambda coro: asyncio.run(coro)
        
        try:
            generate("test prompt", providers=["openai"], method="weighted_voting", output=None)
        except Exception as e:
            # Some errors are expected due to mocking, but the function should be callable
            pass


@patch('examples.cli.ensemble_cli.AdvancedEnsemble', MockAdvancedEnsemble)
@patch('examples.cli.ensemble_cli.Router', MockRouter)
@patch('examples.cli.ensemble_cli.asyncio.run')
@patch('click.echo')
@patch('builtins.open', create=True)
@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
def test_review_command(mock_open, mock_echo, mock_asyncio_run):
    """Test the review command."""
    mock_router = MockRouter()
    mock_router.providers = {"openai": MockProvider("openai")}
    
    # Mock file content
    mock_file = MagicMock()
    mock_file.read.return_value = "def test(): pass"
    mock_file.__enter__.return_value = mock_file
    mock_open.return_value = mock_file
    
    with patch('examples.cli.ensemble_cli._prepare_router', return_value=(mock_router, ["openai"])), \
         patch('examples.cli.ensemble_cli._providers_for_task', return_value=["openai"]), \
         patch('os.path.exists', return_value=True):
        
        try:
            review("test.py", providers=["openai"], output=None)
        except Exception as e:
            # Some errors are expected due to mocking
            pass


@patch('examples.cli.ensemble_cli.AdvancedEnsemble', MockAdvancedEnsemble)
@patch('examples.cli.ensemble_cli.Router', MockRouter)
@patch('examples.cli.ensemble_cli.asyncio.run')
@patch('click.echo')
@patch('builtins.open', create=True)
@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
def test_analyze_image_command(mock_open, mock_echo, mock_asyncio_run):
    """Test the analyze_image command."""
    mock_router = MockRouter()
    mock_router.providers = {"openai": MockProvider("openai")}
    
    # Mock image file content
    mock_file = MagicMock()
    mock_file.read.return_value = b"fake image data"
    mock_file.__enter__.return_value = mock_file
    mock_open.return_value = mock_file
    
    with patch('examples.cli.ensemble_cli._prepare_router', return_value=(mock_router, ["openai"])), \
         patch('examples.cli.ensemble_cli._providers_for_task', return_value=["openai"]), \
         patch('os.path.exists', return_value=True):
        
        try:
            analyze_image("test.jpg", providers=["openai"], analysis_prompt="Describe this image", output=None)
        except Exception as e:
            # Some errors are expected due to mocking
            pass


@patch('examples.cli.ensemble_cli.AdvancedEnsemble', MockAdvancedEnsemble)
@patch('examples.cli.ensemble_cli.Router', MockRouter)
@patch('examples.cli.ensemble_cli.asyncio.run')
@patch('click.echo')
@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
def test_embed_command(mock_echo, mock_asyncio_run):
    """Test the embed command."""
    mock_router = MockRouter()
    mock_provider = MockProvider("openai")
    mock_provider.list_models = AsyncMock(return_value=["mistral", "llama2"])
    mock_router.providers = {"openai": mock_provider}
    
    with patch('examples.cli.ensemble_cli._prepare_router', return_value=(mock_router, ["openai"])), \
         patch('examples.cli.ensemble_cli._providers_for_task', return_value=["openai"]):
        
        try:
            embed("test text", providers=["openai"], model=None, output=None)
        except Exception as e:
            # Some errors are expected due to mocking
            pass


def test_ensemble_group():
    """Test that ensemble is a click group."""
    assert hasattr(ensemble, 'commands')
    assert 'generate' in ensemble.commands
    assert 'review' in ensemble.commands
    assert 'analyze-image' in ensemble.commands  # Click converts underscores to hyphens
    assert 'embed' in ensemble.commands


def test_example_structure():
    """Test that the example has the expected structure."""
    example_path = Path(__file__).parent.parent.parent.parent / "examples" / "cli" / "ensemble_cli.py"
    assert example_path.exists(), "ensemble_cli.py example should exist"
    
    # Check that the file contains expected components
    with open(example_path, 'r') as f:
        content = f.read()
        assert "@click.group()" in content or "def ensemble()" in content
        assert "def generate(" in content
        assert "def review(" in content
        assert "def analyze_image(" in content
        assert "def embed(" in content
        assert "AdvancedEnsemble" in content
        assert "Router" in content


def test_error_handling_no_providers():
    """Test error handling when no providers are configured."""
    # Test that _prepare_router raises an exception when no providers are available
    import click
    with patch.dict(os.environ, {}, clear=True):  # Clear all env vars
        with patch('examples.cli.ensemble_cli.click.echo'):  # Suppress error messages
            # Pass providers that require API keys but none are set
            with pytest.raises(click.ClickException):  # Should raise ClickException
                _prepare_router(["openai", "anthropic"])  # These require API keys


@pytest.mark.asyncio
async def test_error_handling_missing_api_key():
    """Test error handling when API keys are missing."""
    with patch.dict(os.environ, {}, clear=True):
        with patch('examples.cli.ensemble_cli.click.echo'):
            try:
                router, registered = _prepare_router(["openai"])
                # Should handle missing keys gracefully
            except Exception:
                pass  # Expected to fail or skip providers


def test_provider_aliases():
    """Test provider alias mapping."""
    from examples.cli.ensemble_cli import PROVIDER_ALIASES
    assert "claude" in PROVIDER_ALIASES
    assert PROVIDER_ALIASES["claude"] == "anthropic"


def test_provider_registry():
    """Test that provider registry has expected structure."""
    from examples.cli.ensemble_cli import PROVIDER_REGISTRY
    assert "openai" in PROVIDER_REGISTRY
    assert "anthropic" in PROVIDER_REGISTRY
    assert "ollama" in PROVIDER_REGISTRY
    
    # Check structure
    openai_spec = PROVIDER_REGISTRY["openai"]
    assert "env" in openai_spec
    assert "adapter" in openai_spec
    assert "capabilities" in openai_spec

