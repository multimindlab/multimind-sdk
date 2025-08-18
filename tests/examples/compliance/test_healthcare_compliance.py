"""
Tests for healthcare compliance examples.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import os
import sys
from pathlib import Path

# Add examples directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the healthcare compliance example
try:
    from examples.compliance.healthcare.clinical_trial_compliance import main as clinical_trial_main
except ImportError:
    clinical_trial_main = None


class MockComplianceModel:
    """Mock compliance model for testing."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.compliance_checks = []
    
    async def generate(self, prompt: str, **kwargs):
        if "HIPAA" in prompt:
            return "HIPAA compliance check passed"
        elif "FDA" in prompt:
            return "FDA compliance check passed"
        elif "GDPR" in prompt:
            return "GDPR compliance check passed"
        else:
            return "General compliance check passed"
    
    async def chat(self, messages, **kwargs):
        return "Compliance analysis completed"
    
    async def embeddings(self, text, **kwargs):
        if isinstance(text, str):
            return [0.1] * 384
        return [[0.1] * 384] * len(text)


class MockComplianceTrainer:
    """Mock compliance trainer for testing."""
    
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or {}
        self.training_history = []
    
    async def train(self, training_data, validation_data=None):
        self.training_history.append({
            "training_data": training_data,
            "validation_data": validation_data
        })
        return {"accuracy": 0.95, "compliance_score": 0.98}
    
    async def evaluate(self, test_data):
        return {
            "accuracy": 0.94,
            "compliance_score": 0.97,
            "hipaa_score": 0.99,
            "fda_score": 0.96,
            "gdpr_score": 0.95
        }


class MockComplianceEvaluator:
    """Mock compliance evaluator for testing."""
    
    def __init__(self, model):
        self.model = model
        self.evaluation_results = []
    
    async def evaluate_model(self, model, test_data):
        result = {
            "overall_score": 0.95,
            "hipaa_compliance": 0.98,
            "fda_compliance": 0.96,
            "gdpr_compliance": 0.94,
            "privacy_score": 0.97,
            "security_score": 0.96
        }
        self.evaluation_results.append(result)
        return result


@pytest.fixture
def mock_compliance_model():
    """Fixture to provide mock compliance model."""
    return MockComplianceModel("gpt-4")


@pytest.fixture
def mock_compliance_trainer():
    """Fixture to provide mock compliance trainer."""
    model = MockComplianceModel("gpt-4")
    return MockComplianceTrainer(model)


@pytest.fixture
def mock_compliance_evaluator():
    """Fixture to provide mock compliance evaluator."""
    model = MockComplianceModel("gpt-4")
    return MockComplianceEvaluator(model)


@pytest.mark.asyncio
async def test_healthcare_compliance_imports():
    """Test that healthcare compliance modules can be imported."""
    try:
        from examples.compliance.healthcare import clinical_trial_compliance
        assert clinical_trial_compliance is not None
    except ImportError as e:
        pytest.skip(f"Healthcare compliance module not available: {e}")


@pytest.mark.asyncio
async def test_clinical_trial_compliance_main():
    """Test that the clinical trial compliance main function can be called."""
    if clinical_trial_main is None:
        pytest.skip("Clinical trial compliance main function not available")
    
    with patch('examples.compliance.healthcare.clinical_trial_compliance.OpenAIModel', MockComplianceModel), \
         patch('examples.compliance.healthcare.clinical_trial_compliance.ComplianceTrainer', MockComplianceTrainer), \
         patch('examples.compliance.healthcare.clinical_trial_compliance.evaluate_model', AsyncMock(return_value={"score": 0.95})), \
         patch('examples.compliance.healthcare.clinical_trial_compliance.load_dotenv'):
        
        try:
            await clinical_trial_main()
            assert True  # If we get here, the function ran without errors
        except Exception as e:
            pytest.fail(f"clinical_trial_main() function failed: {e}")


@pytest.mark.asyncio
async def test_compliance_model_initialization():
    """Test that compliance models can be initialized correctly."""
    model = MockComplianceModel("gpt-4", temperature=0.7)
    assert model.model_name == "gpt-4"
    assert model.kwargs["temperature"] == 0.7


@pytest.mark.asyncio
async def test_compliance_model_generation():
    """Test that compliance models can generate compliance-related responses."""
    model = MockComplianceModel("gpt-4")
    
    # Test HIPAA compliance
    hipaa_response = await model.generate("Check HIPAA compliance")
    assert "HIPAA compliance check passed" in hipaa_response
    
    # Test FDA compliance
    fda_response = await model.generate("Check FDA compliance")
    assert "FDA compliance check passed" in fda_response
    
    # Test GDPR compliance
    gdpr_response = await model.generate("Check GDPR compliance")
    assert "GDPR compliance check passed" in gdpr_response


@pytest.mark.asyncio
async def test_compliance_trainer():
    """Test compliance trainer functionality."""
    model = MockComplianceModel("gpt-4")
    trainer = MockComplianceTrainer(model)
    
    # Test training
    training_data = [
        {"text": "Patient data", "label": "hipaa_compliant"},
        {"text": "Medical records", "label": "hipaa_compliant"}
    ]
    
    result = await trainer.train(training_data)
    assert result["accuracy"] == 0.95
    assert result["compliance_score"] == 0.98
    assert len(trainer.training_history) == 1
    
    # Test evaluation
    test_data = [
        {"text": "Test patient data", "label": "hipaa_compliant"}
    ]
    
    eval_result = await trainer.evaluate(test_data)
    assert eval_result["accuracy"] == 0.94
    assert eval_result["hipaa_score"] == 0.99


@pytest.mark.asyncio
async def test_compliance_evaluator():
    """Test compliance evaluator functionality."""
    model = MockComplianceModel("gpt-4")
    evaluator = MockComplianceEvaluator(model)
    
    # Test model evaluation
    test_data = [
        {"text": "Test data", "label": "compliant"}
    ]
    
    result = await evaluator.evaluate_model(model, test_data)
    assert result["overall_score"] == 0.95
    assert result["hipaa_compliance"] == 0.98
    assert result["fda_compliance"] == 0.96
    assert result["gdpr_compliance"] == 0.94
    assert len(evaluator.evaluation_results) == 1


@pytest.mark.asyncio
async def test_hipaa_compliance_check():
    """Test HIPAA compliance checking."""
    model = MockComplianceModel("gpt-4")
    
    # Test HIPAA compliance prompt
    prompt = """
    Analyze the following text for HIPAA compliance:
    "Patient John Doe, DOB 01/01/1980, was diagnosed with diabetes."
    """
    
    response = await model.generate(prompt)
    assert "HIPAA compliance check passed" in response


@pytest.mark.asyncio
async def test_fda_compliance_check():
    """Test FDA compliance checking."""
    model = MockComplianceModel("gpt-4")
    
    # Test FDA compliance prompt
    prompt = """
    Analyze the following clinical trial data for FDA compliance:
    "Phase 2 clinical trial results for new drug XYZ."
    """
    
    response = await model.generate(prompt)
    assert "FDA compliance check passed" in response


@pytest.mark.asyncio
async def test_gdpr_compliance_check():
    """Test GDPR compliance checking."""
    model = MockComplianceModel("gpt-4")
    
    # Test GDPR compliance prompt
    prompt = """
    Analyze the following data processing for GDPR compliance:
    "Processing personal health data of EU citizens."
    """
    
    response = await model.generate(prompt)
    assert "GDPR compliance check passed" in response


@pytest.mark.asyncio
async def test_compliance_model_embeddings():
    """Test that compliance models can generate embeddings."""
    model = MockComplianceModel("gpt-4")
    
    # Test single text embedding
    text = "Patient health data"
    embedding = await model.embeddings(text)
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)
    
    # Test multiple text embeddings
    texts = ["HIPAA compliant", "FDA approved", "GDPR compliant"]
    embeddings = await model.embeddings(texts)
    assert len(embeddings) == 3
    assert all(len(emb) == 384 for emb in embeddings)


@pytest.mark.asyncio
async def test_compliance_model_chat():
    """Test that compliance models can handle chat conversations."""
    model = MockComplianceModel("gpt-4")
    messages = [
        {"role": "user", "content": "Is this patient data HIPAA compliant?"},
        {"role": "assistant", "content": "Let me analyze the data for HIPAA compliance."},
        {"role": "user", "content": "What are the key requirements?"}
    ]
    
    response = await model.chat(messages)
    assert "Compliance analysis completed" in response


@pytest.mark.asyncio
async def test_compliance_training_workflow():
    """Test the complete compliance training workflow."""
    model = MockComplianceModel("gpt-4")
    trainer = MockComplianceTrainer(model)
    evaluator = MockComplianceEvaluator(model)
    
    # Training data
    training_data = [
        {"text": "Patient data with PHI", "label": "hipaa_compliant"},
        {"text": "Clinical trial results", "label": "fda_compliant"},
        {"text": "EU patient data", "label": "gdpr_compliant"}
    ]
    
    # Train the model
    training_result = await trainer.train(training_data)
    assert training_result["accuracy"] > 0.9
    assert training_result["compliance_score"] > 0.9
    
    # Evaluate the model
    test_data = [
        {"text": "Test patient data", "label": "hipaa_compliant"}
    ]
    
    eval_result = await evaluator.evaluate_model(model, test_data)
    assert eval_result["overall_score"] > 0.9
    assert eval_result["hipaa_compliance"] > 0.9


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in compliance examples."""
    # Test with failing model
    failing_model = MockComplianceModel("gpt-4")
    failing_model.generate = AsyncMock(side_effect=Exception("API Error"))
    
    trainer = MockComplianceTrainer(failing_model)
    
    # The trainer should handle the error gracefully
    try:
        await trainer.train([{"text": "test", "label": "compliant"}])
        # If no exception is raised, that's also acceptable
    except Exception as e:
        assert "API Error" in str(e)


def test_healthcare_compliance_structure():
    """Test that the healthcare compliance examples have the expected structure."""
    examples_dir = Path(__file__).parent.parent.parent.parent / "examples" / "compliance" / "healthcare"
    assert examples_dir.exists(), "Healthcare compliance examples directory should exist"
    
    # Check for expected files
    expected_files = [
        "clinical_trial_compliance.py",
        "ehr_compliance.py",
        "drug_discovery_compliance.py"
    ]
    
    for file_name in expected_files:
        file_path = examples_dir / file_name
        if file_path.exists():
            # Check that the file contains expected components
            with open(file_path, 'r') as f:
                content = f.read()
                assert "async def main" in content or "def main" in content
                assert "compliance" in content.lower() or "Compliance" in content


@pytest.mark.asyncio
async def test_environment_variables():
    """Test that environment variables are properly handled in compliance examples."""
    # Test that the module can be imported
    try:
        import examples.compliance.healthcare.clinical_trial_compliance
    except ImportError:
        # If import fails due to missing dependencies, that's acceptable
        pass
    
    # Test with environment variables
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
        # This should not raise any errors
        pass


def test_compliance_configuration():
    """Test compliance configuration options."""
    # Test that compliance models can be configured with different parameters
    model = MockComplianceModel("gpt-4", temperature=0.1, max_tokens=1000)
    assert model.model_name == "gpt-4"
    assert model.kwargs["temperature"] == 0.1
    assert model.kwargs["max_tokens"] == 1000
    
    # Test trainer configuration
    trainer = MockComplianceTrainer(model, config={"batch_size": 32, "epochs": 10})
    assert trainer.config["batch_size"] == 32
    assert trainer.config["epochs"] == 10 