"""
Tests for the ensemble examples module.
"""

import asyncio
import sys
import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
pytest.importorskip("numpy")  # requires optional extras absent on core-only installs

# Make the examples package importable
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import examples.ensemble.ensemble_examples as ensemble_examples  # noqa: E402
from multimind import TaskType  # noqa: E402
from multimind.core.provider import GenerationResult, EmbeddingResult, ImageAnalysisResult  # noqa: E402
from multimind.ensemble.advanced import EnsembleResult, ConfidenceScore  # noqa: E402


class DummyRouter:
    """Minimal router stub that records providers and returns stubbed results."""

    def __init__(self, providers, fail_provider=None):
        self.providers = {name: object() for name in providers}
        self.fail_provider = fail_provider

    async def route(self, task_type, input_data, **kwargs):
        provider = kwargs.get("provider")
        model = kwargs.get("model")
        if provider == self.fail_provider:
            raise Exception(f"{provider} failed")

        if task_type == TaskType.EMBEDDINGS:
            return EmbeddingResult(
                embedding=[0.1, 0.2, 0.3],
                tokens_used=10,
                provider_name=provider,
                model_name=model or "default",
                latency_ms=1.0,
                cost_estimate_usd=0.0,
                metadata={},
            )

        if task_type == TaskType.IMAGE_ANALYSIS:
            return ImageAnalysisResult(
                text="Image analysis result",
                captions=["caption1", "caption2"],
                objects=[],
                tokens_used=20,
                provider_name=provider,
                model_name=model or "default",
                latency_ms=2.0,
                cost_estimate_usd=0.0,
                metadata={},
            )

        # Text generation stub
        return GenerationResult(
            text="Generated text response",
            tokens_used=5,
            provider_name=provider,
            model_name=model or "default",
            latency_ms=1.0,
            cost_estimate_usd=0.0,
            metadata={},
        )


class DummyEnsemble:
    """Minimal ensemble stub that returns EnsembleResult objects."""

    def __init__(self, router):
        self.router = router

    async def combine_results(self, results, method, task_type, **kwargs):
        # Create a mock EnsembleResult
        result = results[0] if results else None
        confidence = ConfidenceScore(
            score=0.8,
            explanation=f"Mock confidence for {method}",
            metadata={}
        )
        # Extract provider names from results or use weights from kwargs
        if "weights" in kwargs:
            provider_votes = kwargs["weights"]
        else:
            # Extract provider names from results
            provider_names = []
            for r in results:
                provider_name = getattr(r, "provider_name", None) or getattr(r, "provider", "unknown")
                if provider_name not in provider_names:
                    provider_names.append(provider_name)
            provider_votes = {p: 1.0 / len(provider_names) if provider_names else 1.0 for p in provider_names}
        
        return EnsembleResult(
            result=result,
            confidence=confidence,
            provider_votes=provider_votes,
            metadata={"method": str(method), "task_type": str(task_type)}
        )


@pytest.fixture(autouse=True)
def _isolate_register_providers(monkeypatch):
    """
    Prevent real provider registration during tests.
    """
    monkeypatch.setattr(ensemble_examples.EnsembleExamples, "_register_providers", lambda self: None)


@pytest.fixture
def dummy_router():
    return DummyRouter(providers=["openai", "anthropic", "ollama"])


@pytest.fixture
def dummy_router_single_provider():
    """Router with only one provider to test single provider weight calculation."""
    return DummyRouter(providers=["openai"])


@pytest.fixture
def dummy_router_with_failure():
    """Router where Ollama will fail to simulate missing local model."""
    return DummyRouter(providers=["openai", "ollama"], fail_provider="ollama")


@pytest.fixture
def dummy_router_no_providers():
    """Router with no providers to test error handling."""
    return DummyRouter(providers=[])


@pytest.fixture
def patched_ensemble(monkeypatch):
    monkeypatch.setattr(ensemble_examples, "AdvancedEnsemble", DummyEnsemble)


@pytest.mark.asyncio
async def test_imports():
    """Module should import without errors."""
    assert ensemble_examples is not None
    assert hasattr(ensemble_examples, "EnsembleExamples")
    assert hasattr(ensemble_examples, "serialize_ensemble_result")


@pytest.mark.asyncio
async def test_run_text_generation_ensemble(monkeypatch, dummy_router, patched_ensemble):
    """Text generation ensemble returns results for all strategies."""
    ex = ensemble_examples.EnsembleExamples()
    ex.router = dummy_router
    ex.ensemble = DummyEnsemble(dummy_router)

    result = await ex.run_text_generation_ensemble("hello")

    expected_keys = {
        "weighted_voting",
        "confidence_cascade",
        "parallel_voting",
        "majority_voting",
        "rank_based",
    }
    assert set(result.keys()) == expected_keys
    # Ensure each strategy produced an EnsembleResult
    for value in result.values():
        assert isinstance(value, EnsembleResult)
        assert hasattr(value, "result")
        assert hasattr(value, "confidence")


@pytest.mark.asyncio
async def test_run_text_generation_ensemble_single_provider(monkeypatch, dummy_router_single_provider, patched_ensemble):
    """Test weight calculation with single provider (division by zero fix)."""
    ex = ensemble_examples.EnsembleExamples()
    ex.router = dummy_router_single_provider
    ex.ensemble = DummyEnsemble(dummy_router_single_provider)

    result = await ex.run_text_generation_ensemble("hello", providers=["openai"])

    # Should not raise division by zero error
    assert "weighted_voting" in result
    assert isinstance(result["weighted_voting"], EnsembleResult)
    # Weight should be 1.0 for single provider
    assert result["weighted_voting"].provider_votes.get("openai") == 1.0


@pytest.mark.asyncio
async def test_run_text_generation_ensemble_no_providers(monkeypatch, dummy_router_no_providers, patched_ensemble):
    """Test error handling when no providers are available."""
    ex = ensemble_examples.EnsembleExamples()
    ex.router = dummy_router_no_providers
    ex.ensemble = DummyEnsemble(dummy_router_no_providers)

    with pytest.raises(ValueError, match="None of the requested providers"):
        await ex.run_text_generation_ensemble("hello", providers=["openai", "anthropic"])


@pytest.mark.asyncio
async def test_run_text_generation_ensemble_provider_filtering(monkeypatch, dummy_router_single_provider, patched_ensemble):
    """Test that only registered providers are used."""
    ex = ensemble_examples.EnsembleExamples()
    ex.router = dummy_router_single_provider  # Only has "openai"
    ex.ensemble = DummyEnsemble(dummy_router_single_provider)

    # Request multiple providers but only openai is registered
    result = await ex.run_text_generation_ensemble("hello", providers=["openai", "anthropic", "ollama"])

    # Should still work with only available provider
    assert "weighted_voting" in result
    assert isinstance(result["weighted_voting"], EnsembleResult)


@pytest.mark.asyncio
async def test_embedding_ensemble_fallback(monkeypatch, dummy_router_with_failure, patched_ensemble):
    """Embedding ensemble should skip failing providers and still return a combined result."""
    ex = ensemble_examples.EnsembleExamples()
    ex.router = dummy_router_with_failure
    ex.ensemble = DummyEnsemble(dummy_router_with_failure)

    combined = await ex.run_embedding_ensemble("text for embedding")

    assert isinstance(combined, EnsembleResult)
    assert combined.result is not None


@pytest.mark.asyncio
async def test_embedding_ensemble_all_providers_fail(monkeypatch, dummy_router_with_failure, patched_ensemble):
    """Test error handling when all embedding providers fail."""
    ex = ensemble_examples.EnsembleExamples()
    # Create router where all providers fail
    failing_router = DummyRouter(providers=["ollama"], fail_provider="ollama")
    ex.router = failing_router
    ex.ensemble = DummyEnsemble(failing_router)

    with pytest.raises(ValueError, match="All embedding providers failed"):
        await ex.run_embedding_ensemble("text for embedding", providers=["ollama"])


@pytest.mark.asyncio
async def test_embedding_ensemble_single_provider(monkeypatch, dummy_router_single_provider, patched_ensemble):
    """Test embedding ensemble with single provider (weight calculation)."""
    ex = ensemble_examples.EnsembleExamples()
    ex.router = dummy_router_single_provider
    ex.ensemble = DummyEnsemble(dummy_router_single_provider)

    result = await ex.run_embedding_ensemble("text", providers=["openai"])

    assert isinstance(result, EnsembleResult)
    # Weight should be 1.0 for single provider
    assert result.provider_votes.get("openai") == 1.0


@pytest.mark.asyncio
async def test_run_qa_ensemble(monkeypatch, dummy_router, patched_ensemble):
    """Test QA ensemble functionality."""
    ex = ensemble_examples.EnsembleExamples()
    ex.router = dummy_router
    ex.ensemble = DummyEnsemble(dummy_router)

    result = await ex.run_qa_ensemble(
        question="What is the capital?",
        context="Paris is the capital of France."
    )

    assert isinstance(result, EnsembleResult)
    assert result.result is not None
    assert hasattr(result.result, "text")


@pytest.mark.asyncio
async def test_run_code_review_ensemble(monkeypatch, dummy_router, patched_ensemble):
    """Test code review ensemble functionality."""
    ex = ensemble_examples.EnsembleExamples()
    ex.router = dummy_router
    ex.ensemble = DummyEnsemble(dummy_router)

    code = "def hello(): return 'world'"
    result = await ex.run_code_review_ensemble(code)

    assert isinstance(result, EnsembleResult)
    assert result.result is not None
    assert hasattr(result.result, "text")


@pytest.mark.asyncio
async def test_run_image_analysis_ensemble(monkeypatch, dummy_router, patched_ensemble, tmp_path):
    """Test image analysis ensemble functionality."""
    ex = ensemble_examples.EnsembleExamples()
    ex.router = dummy_router
    ex.ensemble = DummyEnsemble(dummy_router)

    # Create a dummy image file
    image_path = tmp_path / "test_image.jpg"
    image_path.write_bytes(b"fake image data")

    result = await ex.run_image_analysis_ensemble(str(image_path))

    assert isinstance(result, EnsembleResult)
    assert result.result is not None
    assert isinstance(result.result, ImageAnalysisResult)


@pytest.mark.asyncio
async def test_run_image_analysis_ensemble_no_providers(monkeypatch, dummy_router_no_providers, patched_ensemble, tmp_path):
    """Test image analysis ensemble error handling."""
    ex = ensemble_examples.EnsembleExamples()
    ex.router = dummy_router_no_providers
    ex.ensemble = DummyEnsemble(dummy_router_no_providers)

    image_path = tmp_path / "test_image.jpg"
    image_path.write_bytes(b"fake image data")

    with pytest.raises(ValueError, match="None of the requested providers"):
        await ex.run_image_analysis_ensemble(str(image_path))


@pytest.mark.asyncio
async def test_serialize_ensemble_result():
    """Test JSON serialization helper function."""
    # Test with EnsembleResult
    result = EnsembleResult(
        result=GenerationResult(
            text="test",
            tokens_used=10,
            provider_name="openai",
            model_name="gpt-4",
            latency_ms=100.0,
            cost_estimate_usd=0.01,
            metadata={}
        ),
        confidence=ConfidenceScore(score=0.9, explanation="test", metadata={}),
        provider_votes={"openai": 1.0},
        metadata={}
    )

    serialized = ensemble_examples.serialize_ensemble_result(result)
    
    # Should be a dictionary
    assert isinstance(serialized, dict)
    assert "result" in serialized
    assert "confidence" in serialized
    assert "provider_votes" in serialized
    
    # Should be JSON serializable
    json_str = json.dumps(serialized, default=str)
    assert isinstance(json_str, str)
    assert "test" in json_str


@pytest.mark.asyncio
async def test_serialize_ensemble_result_dict():
    """Test serialization with nested dictionaries."""
    result_dict = {
        "weighted_voting": EnsembleResult(
            result=GenerationResult(
                text="test1",
                tokens_used=10,
                provider_name="openai",
                model_name="gpt-4",
                latency_ms=100.0,
                cost_estimate_usd=0.01,
                metadata={}
            ),
            confidence=ConfidenceScore(score=0.9, explanation="test", metadata={}),
            provider_votes={"openai": 1.0},
            metadata={}
        ),
        "confidence_cascade": EnsembleResult(
            result=GenerationResult(
                text="test2",
                tokens_used=20,
                provider_name="ollama",
                model_name="mistral",
                latency_ms=200.0,
                cost_estimate_usd=0.0,
                metadata={}
            ),
            confidence=ConfidenceScore(score=0.8, explanation="test2", metadata={}),
            provider_votes={"ollama": 1.0},
            metadata={}
        )
    }

    serialized = ensemble_examples.serialize_ensemble_result(result_dict)
    
    assert isinstance(serialized, dict)
    assert len(serialized) == 2
    assert "weighted_voting" in serialized
    assert "confidence_cascade" in serialized
    
    # Should be JSON serializable
    json_str = json.dumps(serialized, default=str)
    assert isinstance(json_str, str)


@pytest.mark.asyncio
async def test_serialize_ensemble_result_list():
    """Test serialization with lists."""
    results = [
        EnsembleResult(
            result=GenerationResult(
                text="test1",
                tokens_used=10,
                provider_name="openai",
                model_name="gpt-4",
                latency_ms=100.0,
                cost_estimate_usd=0.01,
                metadata={}
            ),
            confidence=ConfidenceScore(score=0.9, explanation="test", metadata={}),
            provider_votes={"openai": 1.0},
            metadata={}
        ),
        EnsembleResult(
            result=GenerationResult(
                text="test2",
                tokens_used=20,
                provider_name="ollama",
                model_name="mistral",
                latency_ms=200.0,
                cost_estimate_usd=0.0,
                metadata={}
            ),
            confidence=ConfidenceScore(score=0.8, explanation="test2", metadata={}),
            provider_votes={"ollama": 1.0},
            metadata={}
        )
    ]

    serialized = ensemble_examples.serialize_ensemble_result(results)
    
    assert isinstance(serialized, list)
    assert len(serialized) == 2
    assert all(isinstance(item, dict) for item in serialized)
    
    # Should be JSON serializable
    json_str = json.dumps(serialized, default=str)
    assert isinstance(json_str, str)


@pytest.mark.asyncio
async def test_weight_calculation_two_providers(monkeypatch, dummy_router, patched_ensemble):
    """Test weight calculation with two providers."""
    ex = ensemble_examples.EnsembleExamples()
    ex.router = dummy_router
    ex.ensemble = DummyEnsemble(dummy_router)

    result = await ex.run_text_generation_ensemble("hello", providers=["openai", "ollama"])

    weighted_result = result["weighted_voting"]
    weights = weighted_result.provider_votes
    
    # Should have weights for both providers
    assert "openai" in weights
    assert "ollama" in weights
    # Weights should sum to approximately 1.0 (allowing for floating point)
    assert abs(sum(weights.values()) - 1.0) < 0.01


@pytest.mark.asyncio
async def test_weight_calculation_three_providers(monkeypatch, dummy_router, patched_ensemble):
    """Test weight calculation with three providers."""
    ex = ensemble_examples.EnsembleExamples()
    ex.router = dummy_router
    ex.ensemble = DummyEnsemble(dummy_router)

    result = await ex.run_text_generation_ensemble("hello", providers=["openai", "anthropic", "ollama"])

    weighted_result = result["weighted_voting"]
    weights = weighted_result.provider_votes
    
    # Should have weights for all three providers
    assert "openai" in weights
    assert "anthropic" in weights
    assert "ollama" in weights
    # Weights should sum to approximately 1.0
    assert abs(sum(weights.values()) - 1.0) < 0.01
    # OpenAI and Anthropic should have higher weights (0.4 each)
    assert weights["openai"] == 0.4
    assert weights["anthropic"] == 0.4
    assert weights["ollama"] == 0.2
