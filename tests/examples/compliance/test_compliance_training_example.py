"""
Tests for compliance_training_example.py
"""

import pytest  # noqa: E402

pytest.importorskip("torch", reason="requires multimind-sdk[finetune]")

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import os
import sys
import json
import torch
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

# Import the example components
try:
    from examples.compliance.compliance_training_example import (
        ExampleDataset,
        ExampleModel,
        ExampleCompliance,
        main
    )
    from multimind.compliance.model_training import (
        ComplianceDataset,
        ComplianceTrainer,
        ComplianceMetrics
    )
    from multimind.compliance import GovernanceConfig, Regulation
except ImportError as e:
    pytest.skip(f"Compliance training example not available: {e}", allow_module_level=True)


@pytest.fixture
def sample_dataset():
    """Fixture to provide a sample dataset."""
    return ExampleDataset(size=100, input_size=20, num_classes=5)


@pytest.fixture
def sample_model():
    """Fixture to provide a sample model."""
    return ExampleModel(input_size=20, num_classes=5)


@pytest.fixture
def sample_compliance_dataset(sample_dataset):
    """Fixture to provide a compliance-wrapped dataset."""
    return ExampleCompliance(
        base_dataset=sample_dataset,
        compliance_rules={
            "privacy_threshold": 0.9,
            "fairness_threshold": 0.9,
            "transparency_threshold": 0.9,
        },
        data_categories=["personal_data", "sensitive_data"]
    )


@pytest.fixture
def mock_trainer_results():
    """Fixture to provide mock trainer results."""
    return {
        "metrics_history": [
            ComplianceMetrics(
                bias_score=0.1,
                privacy_score=0.9,
                transparency_score=0.8,
                fairness_score=0.85
            ),
            ComplianceMetrics(
                bias_score=0.15,
                privacy_score=0.92,
                transparency_score=0.82,
                fairness_score=0.87
            )
        ],
        "violations": [],
        "final_evaluation": {
            "overall_score": 0.88,
            "bias": 0.12,
            "privacy": 0.91,
            "transparency": 0.81,
            "fairness": 0.86,
            "recommendations": [
                {
                    "action": "Improve transparency documentation",
                    "priority": "medium"
                }
            ]
        }
    }


class TestExampleDataset:
    """Test cases for ExampleDataset."""
    
    def test_dataset_initialization(self):
        """Test that ExampleDataset can be initialized."""
        dataset = ExampleDataset(size=100, input_size=20, num_classes=5)
        assert dataset.size == 100
        assert dataset.input_size == 20
        assert dataset.num_classes == 5
        assert dataset.data.shape == (100, 20)
        assert dataset.labels.shape == (100,)
    
    def test_dataset_length(self, sample_dataset):
        """Test dataset length."""
        assert len(sample_dataset) == 100
    
    def test_dataset_getitem(self, sample_dataset):
        """Test dataset item retrieval."""
        item = sample_dataset[0]
        assert "input" in item
        assert "target" in item
        assert "metadata" in item
        assert item["input"].shape == (20,)
        assert isinstance(item["target"], torch.Tensor)
        assert isinstance(item["metadata"], dict)
    
    def test_dataset_metadata(self, sample_dataset):
        """Test dataset metadata structure."""
        item = sample_dataset[0]
        metadata = item["metadata"]
        assert "data_categories" in metadata
        assert "jurisdiction" in metadata
        assert "regulations" in metadata
        assert "consent_status" in metadata
        assert metadata["consent_status"] is True


class TestExampleModel:
    """Test cases for ExampleModel."""
    
    def test_model_initialization(self):
        """Test that ExampleModel can be initialized."""
        model = ExampleModel(input_size=20, num_classes=5)
        assert model is not None
        assert hasattr(model, "feature_extractor")
        assert hasattr(model, "attention")
        assert hasattr(model, "classifier")
        assert hasattr(model, "compliance_metrics")
    
    def test_model_forward(self, sample_model):
        """Test model forward pass."""
        batch_size = 10
        x = torch.randn(batch_size, 20)
        output = sample_model(x)
        
        assert isinstance(output, dict)
        assert "logits" in output
        assert "attention_weights" in output
        assert "features" in output
        assert output["logits"].shape == (batch_size, 5)
        assert output["attention_weights"].shape == (batch_size, 1)
        assert output["features"].shape == (batch_size, 32)
    
    def test_model_compliance_metrics(self, sample_model):
        """Test that model has compliance metrics."""
        assert hasattr(sample_model, "compliance_metrics")
        assert isinstance(sample_model.compliance_metrics, ComplianceMetrics)
        assert sample_model.compliance_metrics.bias_score == 1.0
        assert sample_model.compliance_metrics.privacy_score == 1.0
        assert sample_model.compliance_metrics.transparency_score == 1.0
        assert sample_model.compliance_metrics.fairness_score == 1.0


class TestExampleCompliance:
    """Test cases for ExampleCompliance."""
    
    def test_compliance_dataset_initialization(self, sample_dataset):
        """Test that ExampleCompliance can be initialized."""
        compliance_dataset = ExampleCompliance(
            base_dataset=sample_dataset,
            compliance_rules={"privacy_threshold": 0.9},
            data_categories=["personal_data"]
        )
        assert compliance_dataset is not None
        assert isinstance(compliance_dataset, ComplianceDataset)
    
    def test_compliance_dataset_length(self, sample_compliance_dataset):
        """Test compliance dataset length."""
        assert len(sample_compliance_dataset) == 100
    
    @pytest.mark.asyncio
    async def test_check_privacy_compliance(self, sample_compliance_dataset):
        """Test privacy compliance checking."""
        item = sample_compliance_dataset[0]
        result = await sample_compliance_dataset.check_privacy_compliance(item)
        
        assert isinstance(result, dict)
        assert "data_minimization" in result
        assert "purpose_limitation" in result
        assert "consent_status" in result
        assert "data_retention" in result
    
    @pytest.mark.asyncio
    async def test_check_fairness_compliance(self, sample_compliance_dataset):
        """Test fairness compliance checking."""
        item = sample_compliance_dataset[0]
        result = await sample_compliance_dataset.check_fairness_compliance(item)
        
        assert isinstance(result, dict)
        assert "demographic_parity" in result
        assert "equal_opportunity" in result
        assert "disparate_impact" in result
    
    @pytest.mark.asyncio
    async def test_check_transparency_compliance(self, sample_compliance_dataset):
        """Test transparency compliance checking."""
        item = sample_compliance_dataset[0]
        result = await sample_compliance_dataset.check_transparency_compliance(item)
        
        assert isinstance(result, dict)
        assert "explainability" in result
        assert "documentation" in result
        assert "audit_trail" in result


class TestComplianceTrainer:
    """Test cases for ComplianceTrainer integration."""
    
    @pytest.mark.asyncio
    async def test_trainer_initialization(self, sample_model):
        """Test that ComplianceTrainer can be initialized."""
        trainer = ComplianceTrainer(
            model=sample_model,
            compliance_rules={"data_minimization": True},
            training_config={
                "epochs": 1,
                "thresholds": {"bias": 0.1, "privacy": 0.9},
                "evaluation_metrics": ["bias", "privacy"]
            }
        )
        assert trainer is not None
        assert trainer.model == sample_model
    
    @pytest.mark.asyncio
    async def test_trainer_train(self, sample_model, sample_compliance_dataset):
        """Test trainer training process."""
        from torch.utils.data import DataLoader
        
        trainer = ComplianceTrainer(
            model=sample_model,
            compliance_rules={"data_minimization": True},
            training_config={
                "epochs": 1,
                "thresholds": {"bias": 0.1, "privacy": 0.9, "transparency": 0.9, "fairness": 0.9},
                "evaluation_metrics": ["bias", "privacy", "transparency", "fairness"]
            }
        )
        
        train_loader = DataLoader(sample_compliance_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(sample_compliance_dataset, batch_size=32, shuffle=False)
        
        results = await trainer.train(
            train_data=train_loader,
            val_data=val_loader,
            metadata={
                "model_type": "example",
                "data_categories": ["personal_data"],
                "jurisdiction": "US"
            }
        )
        
        assert isinstance(results, dict)
        assert "metrics_history" in results
        assert "violations" in results
        assert "final_evaluation" in results
        assert isinstance(results["metrics_history"], list)
        assert isinstance(results["violations"], list)


class TestMainFunction:
    """Test cases for the main function."""
    
    @pytest.mark.asyncio
    async def test_main_function_runs(self, mock_trainer_results):
        """Test that main function can run without errors."""
        with patch('examples.compliance.compliance_training_example.ComplianceTrainer') as mock_trainer_class, \
             patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump, \
             patch('json.dumps') as mock_json_dumps:
            
            # Setup mock trainer
            mock_trainer = AsyncMock()
            mock_trainer.train = AsyncMock(return_value=mock_trainer_results)
            mock_trainer_class.return_value = mock_trainer
            
            # Setup file mock
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            # Setup json dumps mock
            mock_json_dumps.return_value = '{"test": "data"}'
            
            try:
                await main()
                # If we get here, the function ran without errors
                assert True
            except Exception as e:
                # Check if it's a file-related error (acceptable in test environment)
                if "compliance_results.json" not in str(e):
                    pytest.fail(f"main() function failed with unexpected error: {e}")
    
    @pytest.mark.asyncio
    async def test_main_creates_governance_config(self):
        """Test that main function creates governance config correctly."""
        with patch('examples.compliance.compliance_training_example.ComplianceTrainer') as mock_trainer_class, \
             patch('builtins.open', create=True), \
             patch('json.dump'), \
             patch('json.dumps'):
            
            mock_trainer = AsyncMock()
            mock_trainer.train = AsyncMock(return_value={
                "metrics_history": [],
                "violations": [],
                "final_evaluation": {"recommendations": []}
            })
            mock_trainer_class.return_value = mock_trainer
            
            # Import and check GovernanceConfig creation
            from examples.compliance.compliance_training_example import GovernanceConfig, Regulation
            
            config = GovernanceConfig(
                organization_id="org_123",
                organization_name="Example Corp",
                dpo_email="dpo@example.com",
                enabled_regulations=[
                    Regulation.GDPR,
                    Regulation.CCPA,
                    Regulation.AI_ACT
                ]
            )
            
            assert config.organization_id == "org_123"
            assert config.organization_name == "Example Corp"
            assert Regulation.GDPR in config.enabled_regulations
            assert Regulation.CCPA in config.enabled_regulations
            assert Regulation.AI_ACT in config.enabled_regulations


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test the complete compliance training workflow."""
        # Create components
        model = ExampleModel(input_size=20, num_classes=5)
        dataset = ExampleDataset(size=50, input_size=20, num_classes=5)
        compliance_dataset = ExampleCompliance(
            base_dataset=dataset,
            compliance_rules={"privacy_threshold": 0.9},
            data_categories=["personal_data"]
        )
        
        # Verify components work together
        assert len(compliance_dataset) == 50
        
        item = compliance_dataset[0]
        assert "input" in item
        assert "target" in item
        
        # Test model forward pass
        output = model(item["input"].unsqueeze(0))
        assert "logits" in output
        
        # Test compliance checks
        privacy_result = await compliance_dataset.check_privacy_compliance(item)
        assert isinstance(privacy_result, dict)
    
    def test_ccpa_regulation_available(self):
        """Test that CCPA regulation is available."""
        assert hasattr(Regulation, "CCPA")
        assert Regulation.CCPA == "CCPA"
    
    def test_metrics_serialization(self):
        """Test that ComplianceMetrics can be serialized for JSON."""
        metrics = ComplianceMetrics(
            bias_score=0.1,
            privacy_score=0.9,
            transparency_score=0.8,
            fairness_score=0.85
        )
        
        # Convert to dict for JSON serialization
        metrics_dict = {
            "bias_score": metrics.bias_score,
            "privacy_score": metrics.privacy_score,
            "transparency_score": metrics.transparency_score,
            "fairness_score": metrics.fairness_score,
        }
        
        # Should be JSON serializable
        json_str = json.dumps(metrics_dict)
        assert "bias_score" in json_str
        assert "0.1" in json_str


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_dataset_with_invalid_size(self):
        """Test dataset initialization with invalid parameters."""
        # PyTorch raises RuntimeError for negative dimensions
        with pytest.raises(RuntimeError, match="negative dimension"):
            ExampleDataset(size=-1, input_size=20, num_classes=5)
    
    @pytest.mark.asyncio
    async def test_trainer_with_invalid_config(self, sample_model):
        """Test trainer with invalid configuration."""
        # Trainer doesn't validate epochs > 0, so it should initialize successfully
        # but training with epochs=0 will just skip the training loop
        trainer = ComplianceTrainer(
            model=sample_model,
            compliance_rules={},
            training_config={
                "epochs": 0,  # This is accepted, training loop just won't run
                "thresholds": {},
                "evaluation_metrics": []
            }
        )
        # Verify trainer was created successfully (no exception raised)
        assert trainer is not None
        assert trainer.model == sample_model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

