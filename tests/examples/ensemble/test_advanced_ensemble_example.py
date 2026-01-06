"""
Tests for advanced_ensemble_example.py example.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multimind.core.provider import GenerationResult
from multimind.core.router import Router, TaskType
from multimind.ensemble.advanced import AdvancedEnsemble, EnsembleMethod


class MockProvider:
    """Mock provider for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.provider_name = name
    
    async def generate_text(self, prompt: str, model: str = None, **kwargs):
        """Mock text generation."""
        return GenerationResult(
            text=f"Mock {self.name} response to: {prompt}",
            tokens_used=100,
            provider_name=self.name,
            model_name=model or "default-model",
            latency_ms=100.0,
            cost_estimate_usd=0.001
        )
    
    async def generate_embeddings(self, text: str, model: str = None, **kwargs):
        """Mock embeddings generation."""
        from multimind.core.provider import EmbeddingResult
        return EmbeddingResult(
            embedding=[0.1] * 384,
            tokens_used=50,
            provider_name=self.name,
            model_name=model or "default-model",
            latency_ms=50.0,
            cost_estimate_usd=0.0005
        )
    
    async def analyze_image(self, image_data, model: str = None, **kwargs):
        """Mock image analysis."""
        from multimind.core.provider import ImageAnalysisResult
        return ImageAnalysisResult(
            objects=[],
            captions=[f"Mock {self.name} image caption"],
            text=f"Mock {self.name} image analysis",
            provider_name=self.name,
            model_name=model or "default-model",
            latency_ms=200.0,
            cost_estimate_usd=0.002
        )


class TestAdvancedEnsembleExample:
    """Test suite for advanced ensemble example."""
    
    @pytest.fixture
    def mock_router(self):
        """Create a mock router."""
        router = Router()
        openai_provider = MockProvider("openai")
        claude_provider = MockProvider("claude")
        router.register_provider("openai", openai_provider)
        router.register_provider("claude", claude_provider)
        return router
    
    @pytest.fixture
    def ensemble(self, mock_router):
        """Create an advanced ensemble instance."""
        return AdvancedEnsemble(mock_router)
    
    @pytest.mark.asyncio
    async def test_ensemble_with_two_providers(self, ensemble):
        """Test ensemble with 2+ LLMs - full ensemble logic."""
        prompt = "Test prompt"
        
        # Get results from both providers
        results = []
        try:
            openai_result = await ensemble.router.route(
                TaskType.TEXT_GENERATION,
                prompt,
                provider="openai",
                model="gpt-4"
            )
            results.append(openai_result)
        except Exception:
            results.append(None)
        
        try:
            claude_result = await ensemble.router.route(
                TaskType.TEXT_GENERATION,
                prompt,
                provider="claude",
                model="claude-3-sonnet"
            )
            results.append(claude_result)
        except Exception:
            results.append(None)
        
        # Should have 2 valid results
        valid_results = [r for r in results if r is not None]
        assert len(valid_results) == 2
        
        # Test weighted voting
        ensemble_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.WEIGHTED_VOTING,
            task_type=TaskType.TEXT_GENERATION,
            weights={"openai": 0.6, "claude": 0.4}
        )
        
        assert ensemble_result is not None
        assert ensemble_result.result is not None
        assert ensemble_result.result.text is not None
        assert ensemble_result.confidence is not None
        assert ensemble_result.confidence.score > 0
        assert len(ensemble_result.provider_votes) == 2
    
    @pytest.mark.asyncio
    async def test_ensemble_with_one_provider_fallback(self, ensemble):
        """Test ensemble with 1 LLM - fallback router mode."""
        prompt = "Test prompt"
        
        # Get results from providers (simulate one failure)
        results = []
        try:
            openai_result = await ensemble.router.route(
                TaskType.TEXT_GENERATION,
                prompt,
                provider="openai",
                model="gpt-4"
            )
            results.append(openai_result)
        except Exception:
            results.append(None)
        
        # Simulate Claude failure
        results.append(None)
        
        # Should have 1 valid result
        valid_results = [r for r in results if r is not None]
        assert len(valid_results) == 1
        
        # Test that it acts as fallback router
        ensemble_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.WEIGHTED_VOTING,
            task_type=TaskType.TEXT_GENERATION,
            weights={"openai": 0.6, "claude": 0.4}
        )
        
        assert ensemble_result is not None
        assert ensemble_result.result is not None
        assert ensemble_result.result.text is not None
        assert ensemble_result.confidence.score == 1.0
        assert "fallback router mode" in ensemble_result.confidence.explanation.lower()
        assert len(ensemble_result.provider_votes) == 1
        assert "openai" in ensemble_result.provider_votes
    
    @pytest.mark.asyncio
    async def test_ensemble_with_zero_providers_hard_failure(self, ensemble):
        """Test ensemble with 0 LLMs - hard failure."""
        # Simulate both providers failing
        results = [None, None]
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="No valid results provided.*Hard failure"):
            await ensemble.combine_results(
                results=results,
                method=EnsembleMethod.WEIGHTED_VOTING,
                task_type=TaskType.TEXT_GENERATION,
                weights={"openai": 0.6, "claude": 0.4}
            )
    
    @pytest.mark.asyncio
    async def test_weighted_voting_method(self, ensemble):
        """Test weighted voting ensemble method."""
        prompt = "Test prompt"
        
        results = []
        openai_result = await ensemble.router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="openai",
            model="gpt-4"
        )
        results.append(openai_result)
        
        claude_result = await ensemble.router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="claude",
            model="claude-3-sonnet"
        )
        results.append(claude_result)
        
        ensemble_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.WEIGHTED_VOTING,
            task_type=TaskType.TEXT_GENERATION,
            weights={"openai": 0.6, "claude": 0.4}
        )
        
        assert ensemble_result is not None
        assert ensemble_result.result is not None
        assert ensemble_result.confidence is not None
        assert ensemble_result.provider_votes is not None
    
    @pytest.mark.asyncio
    async def test_confidence_cascade_method(self, ensemble):
        """Test confidence cascade ensemble method."""
        prompt = "Test prompt"
        
        results = []
        openai_result = await ensemble.router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="openai",
            model="gpt-4"
        )
        results.append(openai_result)
        
        claude_result = await ensemble.router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="claude",
            model="claude-3-sonnet"
        )
        results.append(claude_result)
        
        ensemble_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.CONFIDENCE_CASCADE,
            task_type=TaskType.TEXT_GENERATION,
            confidence_threshold=0.8
        )
        
        assert ensemble_result is not None
        assert ensemble_result.result is not None
        assert ensemble_result.confidence is not None
    
    @pytest.mark.asyncio
    async def test_parallel_voting_method(self, ensemble):
        """Test parallel voting ensemble method."""
        prompt = "Test prompt"
        
        results = []
        openai_result = await ensemble.router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="openai",
            model="gpt-4"
        )
        results.append(openai_result)
        
        claude_result = await ensemble.router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="claude",
            model="claude-3-sonnet"
        )
        results.append(claude_result)
        
        ensemble_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.PARALLEL_VOTING,
            task_type=TaskType.TEXT_GENERATION
        )
        
        assert ensemble_result is not None
        assert ensemble_result.result is not None
        assert ensemble_result.confidence is not None
    
    @pytest.mark.asyncio
    async def test_majority_voting_method(self, ensemble):
        """Test majority voting ensemble method."""
        prompt = "Test prompt"
        
        results = []
        openai_result = await ensemble.router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="openai",
            model="gpt-4"
        )
        results.append(openai_result)
        
        claude_result = await ensemble.router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="claude",
            model="claude-3-sonnet"
        )
        results.append(claude_result)
        
        ensemble_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.MAJORITY_VOTING,
            task_type=TaskType.TEXT_GENERATION
        )
        
        assert ensemble_result is not None
        assert ensemble_result.result is not None
        assert ensemble_result.confidence is not None
    
    @pytest.mark.asyncio
    async def test_rank_based_method(self, ensemble):
        """Test rank-based ensemble method."""
        prompt = "Test prompt"
        
        results = []
        openai_result = await ensemble.router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="openai",
            model="gpt-4"
        )
        results.append(openai_result)
        
        claude_result = await ensemble.router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="claude",
            model="claude-3-sonnet"
        )
        results.append(claude_result)
        
        ensemble_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.RANK_BASED,
            task_type=TaskType.TEXT_GENERATION
        )
        
        assert ensemble_result is not None
        assert ensemble_result.result is not None
        assert ensemble_result.confidence is not None
    
    @pytest.mark.asyncio
    async def test_router_ensemble_handling_two_providers(self, mock_router):
        """Test router's _handle_ensemble with 2+ providers."""
        from multimind.core.router import TaskConfig, RoutingStrategy
        
        config = TaskConfig(
            preferred_providers=["openai", "claude"],
            fallback_providers=[],
            routing_strategy=RoutingStrategy.ENSEMBLE,
            ensemble_config={"method": "weighted_voting", "weights": {"openai": 0.6, "claude": 0.4}}
        )
        
        mock_router.configure_task(TaskType.TEXT_GENERATION, config)
        
        result = await mock_router.route(
            TaskType.TEXT_GENERATION,
            "Test prompt",
            model="gpt-4"
        )
        
        assert result is not None
        assert result.text is not None
    
    @pytest.mark.asyncio
    async def test_router_ensemble_handling_one_provider(self, mock_router):
        """Test router's _handle_ensemble with 1 provider (fallback mode)."""
        from multimind.core.router import TaskConfig, RoutingStrategy
        
        # Create a router with only one working provider
        router = Router()
        router.register_provider("openai", MockProvider("openai"))
        
        config = TaskConfig(
            preferred_providers=["openai"],
            fallback_providers=[],
            routing_strategy=RoutingStrategy.ENSEMBLE,
            ensemble_config={"method": "weighted_voting", "weights": {"openai": 1.0}}
        )
        
        router.configure_task(TaskType.TEXT_GENERATION, config)
        
        result = await router.route(
            TaskType.TEXT_GENERATION,
            "Test prompt",
            model="gpt-4"
        )
        
        assert result is not None
        assert result.text is not None
    
    @pytest.mark.asyncio
    async def test_router_ensemble_handling_zero_providers(self, mock_router):
        """Test router's _handle_ensemble with 0 providers (hard failure)."""
        from multimind.core.router import TaskConfig, RoutingStrategy
        
        # Create a router with providers that will all fail
        router = Router()
        
        # Create a mock provider that always raises an exception
        class FailingProvider:
            async def generate_text(self, prompt: str, model: str = None, **kwargs):
                raise Exception("Provider failed")
            
            async def generate_embeddings(self, text: str, model: str = None, **kwargs):
                raise Exception("Provider failed")
            
            async def analyze_image(self, image_data, model: str = None, **kwargs):
                raise Exception("Provider failed")
        
        router.register_provider("failing1", FailingProvider())
        router.register_provider("failing2", FailingProvider())
        
        config = TaskConfig(
            preferred_providers=["failing1", "failing2"],
            fallback_providers=[],
            routing_strategy=RoutingStrategy.ENSEMBLE,
            ensemble_config={"method": "weighted_voting", "weights": {"failing1": 0.5, "failing2": 0.5}}
        )
        
        router.configure_task(TaskType.TEXT_GENERATION, config)
        
        # Should raise exception with "failed" in message
        with pytest.raises(Exception, match=".*failed.*"):
            await router.route(
                TaskType.TEXT_GENERATION,
                "Test prompt",
                model="gpt-4"
            )
    
    @pytest.mark.asyncio
    async def test_graceful_failure_handling(self, ensemble):
        """Test that the example handles provider failures gracefully."""
        prompt = "Test prompt"
        
        # Simulate OpenAI success and Claude failure
        results = []
        
        # OpenAI succeeds
        try:
            openai_result = await ensemble.router.route(
                TaskType.TEXT_GENERATION,
                prompt,
                provider="openai",
                model="gpt-4"
            )
            results.append(openai_result)
        except Exception as e:
            results.append(None)
        
        # Claude fails (simulated)
        results.append(None)
        
        # Should not raise exception, should return fallback result
        ensemble_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.WEIGHTED_VOTING,
            task_type=TaskType.TEXT_GENERATION,
            weights={"openai": 0.6, "claude": 0.4}
        )
        
        assert ensemble_result is not None
        assert ensemble_result.result is not None
        assert ensemble_result.result.text is not None
        # Should be in fallback mode
        assert ensemble_result.confidence.score == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

