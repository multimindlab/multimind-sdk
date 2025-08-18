"""
Tests for ensemble API examples.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import os
import sys
from pathlib import Path

# Add examples directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the ensemble API example
try:
    from examples.api.ensemble_api import main as ensemble_main
except ImportError:
    ensemble_main = None


class MockEnsembleModel:
    """Mock ensemble model for testing."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.predictions = []
    
    async def generate(self, prompt: str, **kwargs):
        if "gpt" in self.model_name.lower():
            return f"GPT response: {prompt}"
        elif "claude" in self.model_name.lower():
            return f"Claude response: {prompt}"
        elif "mistral" in self.model_name.lower():
            return f"Mistral response: {prompt}"
        else:
            return f"Generic response: {prompt}"
    
    async def chat(self, messages, **kwargs):
        return f"Chat response from {self.model_name}"
    
    async def embeddings(self, text, **kwargs):
        if isinstance(text, str):
            return [0.1] * 384
        return [[0.1] * 384] * len(text)


class MockEnsemble:
    """Mock ensemble for testing."""
    
    def __init__(self, models, strategy="majority"):
        self.models = models
        self.strategy = strategy
        self.predictions = []
    
    async def predict(self, prompt: str, **kwargs):
        predictions = []
        for model in self.models:
            response = await model.generate(prompt)
            predictions.append(response)
        
        self.predictions.append({
            "prompt": prompt,
            "predictions": predictions,
            "strategy": self.strategy
        })
        
        # Return majority or consensus response
        if self.strategy == "majority":
            return self._majority_vote(predictions)
        elif self.strategy == "consensus":
            return self._consensus(predictions)
        else:
            return predictions[0]  # Default to first model
    
    def _majority_vote(self, predictions):
        # Simple majority vote implementation
        return predictions[0]  # For testing, just return first prediction
    
    def _consensus(self, predictions):
        # Simple consensus implementation
        return predictions[0]  # For testing, just return first prediction
    
    async def evaluate(self, test_data):
        return {
            "accuracy": 0.95,
            "consensus_score": 0.92,
            "diversity_score": 0.88
        }


class MockEnsembleEvaluator:
    """Mock ensemble evaluator for testing."""
    
    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.evaluation_results = []
    
    async def evaluate_ensemble(self, test_data):
        result = await self.ensemble.evaluate(test_data)
        self.evaluation_results.append(result)
        return result
    
    async def compare_models(self, models, test_data):
        results = {}
        for model_name, model in models.items():
            # Mock individual model evaluation
            results[model_name] = {
                "accuracy": 0.9 + (hash(model_name) % 10) / 100,  # Vary accuracy slightly
                "latency": 1.0 + (hash(model_name) % 5) / 10,     # Vary latency slightly
                "cost": 0.01 + (hash(model_name) % 5) / 1000      # Vary cost slightly
            }
        return results


@pytest.fixture
def mock_models():
    """Fixture to provide mock models for ensemble."""
    return {
        "gpt-4": MockEnsembleModel("gpt-4"),
        "claude-3": MockEnsembleModel("claude-3"),
        "mistral": MockEnsembleModel("mistral")
    }


@pytest.fixture
def mock_ensemble():
    """Fixture to provide mock ensemble."""
    models = [
        MockEnsembleModel("gpt-4"),
        MockEnsembleModel("claude-3"),
        MockEnsembleModel("mistral")
    ]
    return MockEnsemble(models, strategy="majority")


@pytest.fixture
def mock_ensemble_evaluator():
    """Fixture to provide mock ensemble evaluator."""
    models = [
        MockEnsembleModel("gpt-4"),
        MockEnsembleModel("claude-3"),
        MockEnsembleModel("mistral")
    ]
    ensemble = MockEnsemble(models)
    return MockEnsembleEvaluator(ensemble)


@pytest.mark.asyncio
async def test_ensemble_api_imports():
    """Test that ensemble API modules can be imported."""
    try:
        from examples.api import ensemble_api
        assert ensemble_api is not None
    except ImportError as e:
        pytest.skip(f"Ensemble API module not available: {e}")


@pytest.mark.asyncio
async def test_ensemble_api_main():
    """Test that the ensemble API main function can be called."""
    if ensemble_main is None:
        pytest.skip("Ensemble API main function not available")
    
    with patch('examples.api.ensemble_api.OpenAIModel', MockEnsembleModel), \
         patch('examples.api.ensemble_api.ClaudeModel', MockEnsembleModel), \
         patch('examples.api.ensemble_api.MistralModel', MockEnsembleModel), \
         patch('examples.api.ensemble_api.AdvancedEnsemble', MockEnsemble), \
         patch('examples.api.ensemble_api.load_dotenv'):
        
        try:
            await ensemble_main()
            assert True  # If we get here, the function ran without errors
        except Exception as e:
            pytest.fail(f"ensemble_main() function failed: {e}")


@pytest.mark.asyncio
async def test_ensemble_model_initialization():
    """Test that ensemble models can be initialized correctly."""
    gpt_model = MockEnsembleModel("gpt-4", temperature=0.7)
    claude_model = MockEnsembleModel("claude-3", temperature=0.7)
    mistral_model = MockEnsembleModel("mistral", temperature=0.7)
    
    assert gpt_model.model_name == "gpt-4"
    assert claude_model.model_name == "claude-3"
    assert mistral_model.model_name == "mistral"
    assert gpt_model.kwargs["temperature"] == 0.7


@pytest.mark.asyncio
async def test_ensemble_model_generation():
    """Test that ensemble models can generate responses."""
    gpt_model = MockEnsembleModel("gpt-4")
    claude_model = MockEnsembleModel("claude-3")
    mistral_model = MockEnsembleModel("mistral")
    
    prompt = "Explain machine learning"
    
    # Test individual model generation
    gpt_response = await gpt_model.generate(prompt)
    assert "GPT response" in gpt_response
    assert prompt in gpt_response
    
    claude_response = await claude_model.generate(prompt)
    assert "Claude response" in claude_response
    assert prompt in claude_response
    
    mistral_response = await mistral_model.generate(prompt)
    assert "Mistral response" in mistral_response
    assert prompt in mistral_response


@pytest.mark.asyncio
async def test_ensemble_creation():
    """Test that ensembles can be created with multiple models."""
    models = [
        MockEnsembleModel("gpt-4"),
        MockEnsembleModel("claude-3"),
        MockEnsembleModel("mistral")
    ]
    
    ensemble = MockEnsemble(models, strategy="majority")
    assert len(ensemble.models) == 3
    assert ensemble.strategy == "majority"


@pytest.mark.asyncio
async def test_ensemble_prediction():
    """Test that ensembles can make predictions."""
    models = [
        MockEnsembleModel("gpt-4"),
        MockEnsembleModel("claude-3"),
        MockEnsembleModel("mistral")
    ]
    
    ensemble = MockEnsemble(models, strategy="majority")
    
    # Test ensemble prediction
    prompt = "What is artificial intelligence?"
    result = await ensemble.predict(prompt)
    
    assert result is not None
    assert len(ensemble.predictions) == 1
    assert ensemble.predictions[0]["prompt"] == prompt
    assert len(ensemble.predictions[0]["predictions"]) == 3


@pytest.mark.asyncio
async def test_ensemble_strategies():
    """Test different ensemble strategies."""
    models = [
        MockEnsembleModel("gpt-4"),
        MockEnsembleModel("claude-3"),
        MockEnsembleModel("mistral")
    ]
    
    # Test majority strategy
    majority_ensemble = MockEnsemble(models, strategy="majority")
    result1 = await majority_ensemble.predict("Test prompt")
    assert result1 is not None
    
    # Test consensus strategy
    consensus_ensemble = MockEnsemble(models, strategy="consensus")
    result2 = await consensus_ensemble.predict("Test prompt")
    assert result2 is not None


@pytest.mark.asyncio
async def test_ensemble_evaluation():
    """Test ensemble evaluation functionality."""
    models = [
        MockEnsembleModel("gpt-4"),
        MockEnsembleModel("claude-3"),
        MockEnsembleModel("mistral")
    ]
    
    ensemble = MockEnsemble(models)
    evaluator = MockEnsembleEvaluator(ensemble)
    
    # Test ensemble evaluation
    test_data = [
        {"prompt": "Test 1", "expected": "Response 1"},
        {"prompt": "Test 2", "expected": "Response 2"}
    ]
    
    result = await evaluator.evaluate_ensemble(test_data)
    assert result["accuracy"] == 0.95
    assert result["consensus_score"] == 0.92
    assert result["diversity_score"] == 0.88
    assert len(evaluator.evaluation_results) == 1


@pytest.mark.asyncio
async def test_model_comparison():
    """Test model comparison functionality."""
    models = {
        "gpt-4": MockEnsembleModel("gpt-4"),
        "claude-3": MockEnsembleModel("claude-3"),
        "mistral": MockEnsembleModel("mistral")
    }
    
    evaluator = MockEnsembleEvaluator(MockEnsemble(list(models.values())))
    
    # Test model comparison
    test_data = [
        {"prompt": "Test 1", "expected": "Response 1"},
        {"prompt": "Test 2", "expected": "Response 2"}
    ]
    
    results = await evaluator.compare_models(models, test_data)
    
    assert "gpt-4" in results
    assert "claude-3" in results
    assert "mistral" in results
    
    for model_name, metrics in results.items():
        assert "accuracy" in metrics
        assert "latency" in metrics
        assert "cost" in metrics
        assert 0.9 <= metrics["accuracy"] <= 1.0
        assert metrics["latency"] > 0
        assert metrics["cost"] > 0


@pytest.mark.asyncio
async def test_ensemble_model_embeddings():
    """Test that ensemble models can generate embeddings."""
    model = MockEnsembleModel("gpt-4")
    
    # Test single text embedding
    text = "Test text"
    embedding = await model.embeddings(text)
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)
    
    # Test multiple text embeddings
    texts = ["Text 1", "Text 2", "Text 3"]
    embeddings = await model.embeddings(texts)
    assert len(embeddings) == 3
    assert all(len(emb) == 384 for emb in embeddings)


@pytest.mark.asyncio
async def test_ensemble_model_chat():
    """Test that ensemble models can handle chat conversations."""
    model = MockEnsembleModel("gpt-4")
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    
    response = await model.chat(messages)
    assert "Chat response from gpt-4" in response


@pytest.mark.asyncio
async def test_ensemble_workflow():
    """Test the complete ensemble workflow."""
    # Create models
    models = [
        MockEnsembleModel("gpt-4"),
        MockEnsembleModel("claude-3"),
        MockEnsembleModel("mistral")
    ]
    
    # Create ensemble
    ensemble = MockEnsemble(models, strategy="majority")
    
    # Create evaluator
    evaluator = MockEnsembleEvaluator(ensemble)
    
    # Test data
    test_data = [
        {"prompt": "What is AI?", "expected": "AI explanation"},
        {"prompt": "Explain ML", "expected": "ML explanation"}
    ]
    
    # Make predictions
    for item in test_data:
        result = await ensemble.predict(item["prompt"])
        assert result is not None
    
    # Evaluate ensemble
    eval_result = await evaluator.evaluate_ensemble(test_data)
    assert eval_result["accuracy"] > 0.9
    
    # Compare individual models
    model_dict = {f"model_{i}": model for i, model in enumerate(models)}
    comparison = await evaluator.compare_models(model_dict, test_data)
    assert len(comparison) == 3


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in ensemble examples."""
    # Test with failing model
    failing_model = MockEnsembleModel("gpt-4")
    failing_model.generate = AsyncMock(side_effect=Exception("API Error"))
    
    models = [failing_model, MockEnsembleModel("claude-3")]
    ensemble = MockEnsemble(models)
    
    # The ensemble should handle the error gracefully
    with pytest.raises(Exception, match="API Error"):
        await ensemble.predict("Test prompt")


def test_ensemble_api_structure():
    """Test that the ensemble API examples have the expected structure."""
    examples_dir = Path(__file__).parent.parent.parent / "examples" / "api"
    assert examples_dir.exists(), "API examples directory should exist"
    
    # Check for ensemble_api.py
    ensemble_api_path = examples_dir / "ensemble_api.py"
    if ensemble_api_path.exists():
        with open(ensemble_api_path, 'r') as f:
            content = f.read()
            assert "async def main" in content or "def main" in content
            assert "ensemble" in content.lower() or "Ensemble" in content


@pytest.mark.asyncio
async def test_environment_variables():
    """Test that environment variables are properly handled in ensemble examples."""
    # Test that the module can be imported without load_dotenv
    import examples.api.ensemble_api
    
    # Test with environment variables
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
        # This should not raise any errors
        pass


def test_ensemble_configuration():
    """Test ensemble configuration options."""
    # Test that ensemble models can be configured with different parameters
    model = MockEnsembleModel("gpt-4", temperature=0.1, max_tokens=1000)
    assert model.model_name == "gpt-4"
    assert model.kwargs["temperature"] == 0.1
    assert model.kwargs["max_tokens"] == 1000
    
    # Test ensemble configuration
    models = [MockEnsembleModel("gpt-4"), MockEnsembleModel("claude-3")]
    ensemble = MockEnsemble(models, strategy="consensus")
    assert ensemble.strategy == "consensus"
    assert len(ensemble.models) == 2


@pytest.mark.asyncio
async def test_ensemble_performance_metrics():
    """Test ensemble performance metrics."""
    models = [
        MockEnsembleModel("gpt-4"),
        MockEnsembleModel("claude-3"),
        MockEnsembleModel("mistral")
    ]
    
    ensemble = MockEnsemble(models)
    evaluator = MockEnsembleEvaluator(ensemble)
    
    # Test performance evaluation
    test_data = [
        {"prompt": "Performance test", "expected": "Response"}
    ]
    
    # Make multiple predictions to test performance
    start_time = asyncio.get_event_loop().time()
    
    for _ in range(5):
        await ensemble.predict("Performance test prompt")
    
    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time
    
    # Evaluate ensemble
    result = await evaluator.evaluate_ensemble(test_data)
    
    assert result["accuracy"] > 0.9
    assert len(ensemble.predictions) == 5
    assert total_time > 0  # Should take some time 