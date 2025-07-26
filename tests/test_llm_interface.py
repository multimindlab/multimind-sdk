"""
Test LLM interface functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from multimind.llm.llm_interface import (
    LLMInterface,
    GenerationConfig,
    GenerationResult,
    ErrorHandlingConfig,
    EnsembleStrategy
)
from multimind.models.base import BaseLLM


class MockLLM(BaseLLM):
    """Mock LLM for testing."""
    
    def __init__(self, model_name: str, response: str = "Mock response"):
        super().__init__(model_name)
        self.response = response
    
    async def generate(self, prompt: str, **kwargs) -> str:
        return self.response
    
    async def generate_stream(self, prompt: str, **kwargs):
        yield self.response
    
    async def chat(self, messages, **kwargs) -> str:
        return self.response
    
    async def chat_stream(self, messages, **kwargs):
        yield self.response
    
    async def embeddings(self, text, **kwargs):
        if isinstance(text, str):
            return [0.1, 0.2, 0.3]
        return [[0.1, 0.2, 0.3] for _ in text]


@pytest.mark.asyncio
async def test_llm_interface_basic_generation():
    """Test basic text generation."""
    mock_llm = MockLLM("test-model", "Test response")
    interface = LLMInterface(
        models={"test-model": mock_llm},
        default_model="test-model"
    )
    
    result = await interface.generate("Test prompt")
    
    assert isinstance(result, GenerationResult)
    assert result.text == "Test response"
    assert result.model == "test-model"


@pytest.mark.asyncio
async def test_llm_interface_ensemble_majority():
    """Test ensemble generation with majority strategy."""
    mock_llm1 = MockLLM("model1", "Response A")
    mock_llm2 = MockLLM("model2", "Response A")
    mock_llm3 = MockLLM("model3", "Response B")
    
    interface = LLMInterface(
        models={
            "model1": mock_llm1,
            "model2": mock_llm2,
            "model3": mock_llm3
        },
        default_model="model1",
        ensemble_strategy="majority"
    )
    
    result = await interface.generate_with_ensemble("Test prompt")
    
    assert isinstance(result, GenerationResult)
    assert result.text == "Response A"  # Majority vote


@pytest.mark.asyncio
async def test_llm_interface_error_handling():
    """Test error handling with retry strategy."""
    mock_llm = MockLLM("test-model")
    mock_llm.generate = AsyncMock(side_effect=Exception("Test error"))
    
    interface = LLMInterface(
        models={"test-model": mock_llm},
        default_model="test-model",
        error_config=ErrorHandlingConfig(
            strategy="retry",
            max_retries=2,
            retry_delay=0.1,
            fallback_model=None,
            custom_params={}
        )
    )
    
    with pytest.raises(Exception):
        await interface.generate("Test prompt")


@pytest.mark.asyncio
async def test_llm_interface_metrics():
    """Test metrics collection."""
    mock_llm = MockLLM("test-model", "Test response")
    interface = LLMInterface(
        models={"test-model": mock_llm},
        default_model="test-model"
    )
    
    # Generate some text
    await interface.generate("Test prompt 1")
    await interface.generate("Test prompt 2")
    
    metrics = interface.get_metrics()
    
    assert metrics["total_requests"] == 2
    assert metrics["successful_requests"] == 2
    assert metrics["failed_requests"] == 0
    assert metrics["success_rate"] == 1.0


@pytest.mark.asyncio
async def test_llm_interface_prompt_analysis():
    """Test prompt analysis functionality."""
    mock_llm = MockLLM("test-model")
    interface = LLMInterface(
        models={"test-model": mock_llm},
        default_model="test-model"
    )
    
    # Test code generation prompt
    analysis = await interface.prompting.analyze_prompt("def sort_list(lst):\n    return sorted(lst)")
    assert analysis["task_type"] == "code_generation"
    assert analysis["has_code"] == True
    
    # Test question answering prompt
    analysis = await interface.prompting.analyze_prompt("What is the capital of France?")
    assert analysis["task_type"] == "question_answering"
    assert analysis["has_questions"] == True


if __name__ == "__main__":
    pytest.main([__file__])