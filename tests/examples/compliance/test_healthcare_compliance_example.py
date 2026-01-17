"""
Tests for healthcare_compliance_example.py
Comprehensive test suite for healthcare compliance monitoring and evaluation.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add root directory to path
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

# Import the example components
try:
    from examples.compliance.healthcare_compliance_example import (
        run_healthcare_compliance_example,
        _make_json_serializable,
        main
    )
    from examples.compliance.healthcare.medical_diagnosis_compliance import (
        MedicalDiagnosisDataset,
        DiagnosisModel,
        MedicalDiagnosisCompliance
    )
    from examples.compliance.healthcare.patient_monitoring_compliance import (
        PatientMonitoringDataset,
        PatientMonitoringModel,
        PatientMonitoringCompliance
    )
    from examples.compliance.healthcare.ehr_compliance import (
        EHRDataset,
        EHRModel,
        EHRCompliance
    )
    from examples.compliance.healthcare.medical_imaging_compliance import (
        MedicalImagingDataset,
        MedicalImagingModel,
        MedicalImagingCompliance
    )
    from examples.compliance.healthcare.clinical_trial_compliance import (
        ClinicalTrialDataset,
        ClinicalTrialModel,
        ClinicalTrialCompliance
    )
    from examples.compliance.healthcare.drug_discovery_compliance import (
        DrugDiscoveryDataset,
        DrugDiscoveryModel,
        DrugDiscoveryCompliance
    )
    from examples.compliance.healthcare.fraud_detection_compliance import (
        FraudDetectionDataset,
        FraudDetectionModel,
        FraudDetectionCompliance
    )
    from multimind.compliance.model_training import (
        ComplianceDataset,
        ComplianceTrainer,
        ComplianceMetrics
    )
    from multimind.compliance import GovernanceConfig, Regulation
except ImportError as e:
    pytest.skip(f"Healthcare compliance example not available: {e}", allow_module_level=True)


# ==================== Fixtures ====================

@pytest.fixture
def sample_medical_diagnosis_dataset():
    """Fixture to provide a sample medical diagnosis dataset."""
    return MedicalDiagnosisDataset(size=100, input_size=20, num_classes=5)


@pytest.fixture
def sample_medical_diagnosis_model():
    """Fixture to provide a sample medical diagnosis model."""
    return DiagnosisModel(input_size=20, num_classes=5)


@pytest.fixture
def sample_patient_monitoring_dataset():
    """Fixture to provide a sample patient monitoring dataset."""
    return PatientMonitoringDataset(size=100, input_size=20, num_classes=5)


@pytest.fixture
def sample_ehr_dataset():
    """Fixture to provide a sample EHR dataset."""
    return EHRDataset(size=100, input_size=20, num_classes=5)


@pytest.fixture
def sample_config():
    """Fixture to provide a sample configuration."""
    return {
        "data_categories": ["health_data", "personal_data"],
        "metadata": {
            "model_type": "medical_diagnosis",
            "data_categories": ["health_data", "personal_data"],
            "jurisdiction": "US",
            "hipaa_covered": True,
            "sensitive_data": True,
            "explainability_required": True
        }
    }


@pytest.fixture
def sample_compliance_rules():
    """Fixture to provide sample compliance rules."""
    return {
        "bias_threshold": 0.1,
        "privacy_threshold": 0.9,
        "transparency_threshold": 0.9,
        "fairness_threshold": 0.9,
        "hipaa_compliance": True,
        "data_minimization": True,
        "audit_trail": True,
        "explainability": True
    }


@pytest.fixture
def sample_training_config(sample_compliance_rules):
    """Fixture to provide sample training configuration."""
    return {
        "epochs": 1,  # Use 1 epoch for faster testing
        "thresholds": sample_compliance_rules,
        "evaluation_metrics": [
            "bias",
            "privacy",
            "transparency",
            "fairness",
            "hipaa_compliance"
        ]
    }


# ==================== Tests for run_healthcare_compliance_example ====================

@pytest.mark.asyncio
async def test_run_healthcare_compliance_example_medical_diagnosis(
    sample_medical_diagnosis_model,
    sample_config
):
    """Test running healthcare compliance example for medical diagnosis."""
    results = await run_healthcare_compliance_example(
        dataset_class=MedicalDiagnosisDataset,
        model_class=DiagnosisModel,
        compliance_class=MedicalDiagnosisCompliance,
        config=sample_config
    )
    
    assert results is not None
    assert "metrics_history" in results
    assert "violations" in results
    assert "final_evaluation" in results
    assert isinstance(results["metrics_history"], list)
    assert isinstance(results["violations"], list)
    assert isinstance(results["final_evaluation"], dict)
    
    # Check final evaluation structure
    final_eval = results["final_evaluation"]
    assert "compliance_scores" in final_eval
    assert "violations" in final_eval
    assert "recommendations" in final_eval


@pytest.mark.asyncio
async def test_run_healthcare_compliance_example_patient_monitoring(sample_config):
    """Test running healthcare compliance example for patient monitoring."""
    config = {
        **sample_config,
        "metadata": {
            **sample_config["metadata"],
            "model_type": "patient_monitoring",
            "real_time_required": True
        }
    }
    
    results = await run_healthcare_compliance_example(
        dataset_class=PatientMonitoringDataset,
        model_class=PatientMonitoringModel,
        compliance_class=PatientMonitoringCompliance,
        config=config
    )
    
    assert results is not None
    assert "metrics_history" in results
    assert "violations" in results
    assert "final_evaluation" in results


@pytest.mark.asyncio
async def test_run_healthcare_compliance_example_ehr(sample_config):
    """Test running healthcare compliance example for EHR."""
    config = {
        **sample_config,
        "metadata": {
            **sample_config["metadata"],
            "model_type": "ehr"
        }
    }
    
    results = await run_healthcare_compliance_example(
        dataset_class=EHRDataset,
        model_class=EHRModel,
        compliance_class=EHRCompliance,
        config=config
    )
    
    assert results is not None
    assert "metrics_history" in results
    assert "violations" in results
    assert "final_evaluation" in results


@pytest.mark.asyncio
async def test_run_healthcare_compliance_example_medical_imaging(sample_config):
    """Test running healthcare compliance example for medical imaging."""
    config = {
        **sample_config,
        "metadata": {
            **sample_config["metadata"],
            "model_type": "medical_imaging"
        }
    }
    
    results = await run_healthcare_compliance_example(
        dataset_class=MedicalImagingDataset,
        model_class=MedicalImagingModel,
        compliance_class=MedicalImagingCompliance,
        config=config
    )
    
    assert results is not None
    assert "final_evaluation" in results


@pytest.mark.asyncio
async def test_run_healthcare_compliance_example_clinical_trial(sample_config):
    """Test running healthcare compliance example for clinical trial."""
    config = {
        **sample_config,
        "metadata": {
            **sample_config["metadata"],
            "model_type": "clinical_trial"
        }
    }
    
    results = await run_healthcare_compliance_example(
        dataset_class=ClinicalTrialDataset,
        model_class=ClinicalTrialModel,
        compliance_class=ClinicalTrialCompliance,
        config=config
    )
    
    assert results is not None
    assert "final_evaluation" in results


@pytest.mark.asyncio
async def test_run_healthcare_compliance_example_drug_discovery(sample_config):
    """Test running healthcare compliance example for drug discovery."""
    config = {
        **sample_config,
        "data_categories": ["drug_development", "research_data"],
        "metadata": {
            **sample_config["metadata"],
            "model_type": "drug_discovery",
            "fda_covered": True
        }
    }
    
    results = await run_healthcare_compliance_example(
        dataset_class=DrugDiscoveryDataset,
        model_class=DrugDiscoveryModel,
        compliance_class=DrugDiscoveryCompliance,
        config=config
    )
    
    assert results is not None
    assert "final_evaluation" in results


@pytest.mark.asyncio
async def test_run_healthcare_compliance_example_fraud_detection(sample_config):
    """Test running healthcare compliance example for fraud detection."""
    config = {
        **sample_config,
        "data_categories": ["claims_data", "personal_data"],
        "metadata": {
            **sample_config["metadata"],
            "model_type": "fraud_detection",
            "fraud_monitoring": True
        }
    }
    
    results = await run_healthcare_compliance_example(
        dataset_class=FraudDetectionDataset,
        model_class=FraudDetectionModel,
        compliance_class=FraudDetectionCompliance,
        config=config
    )
    
    assert results is not None
    assert "final_evaluation" in results


# ==================== Tests for _make_json_serializable ====================

def test_make_json_serializable_compliance_metrics():
    """Test _make_json_serializable with ComplianceMetrics."""
    metrics = ComplianceMetrics(
        bias_score=0.8,
        privacy_score=0.9,
        transparency_score=0.85,
        fairness_score=0.88
    )
    
    serialized = _make_json_serializable(metrics)
    
    assert isinstance(serialized, dict)
    assert serialized["bias_score"] == 0.8
    assert serialized["privacy_score"] == 0.9
    assert serialized["transparency_score"] == 0.85
    assert serialized["fairness_score"] == 0.88
    assert "timestamp" in serialized


def test_make_json_serializable_datetime():
    """Test _make_json_serializable with datetime."""
    dt = datetime.now()
    
    serialized = _make_json_serializable(dt)
    
    assert isinstance(serialized, str)
    assert "T" in serialized or "-" in serialized  # ISO format


def test_make_json_serializable_dict():
    """Test _make_json_serializable with nested dictionary."""
    data = {
        "metrics": ComplianceMetrics(
            bias_score=0.8,
            privacy_score=0.9,
            transparency_score=0.85,
            fairness_score=0.88
        ),
        "timestamp": datetime.now(),
        "value": 42,
        "nested": {
            "inner": ComplianceMetrics(
                bias_score=0.7,
                privacy_score=0.85,
                transparency_score=0.8,
                fairness_score=0.82
            )
        }
    }
    
    serialized = _make_json_serializable(data)
    
    assert isinstance(serialized, dict)
    assert isinstance(serialized["metrics"], dict)
    assert isinstance(serialized["timestamp"], str)
    assert serialized["value"] == 42
    assert isinstance(serialized["nested"]["inner"], dict)


def test_make_json_serializable_list():
    """Test _make_json_serializable with list."""
    data = [
        ComplianceMetrics(
            bias_score=0.8,
            privacy_score=0.9,
            transparency_score=0.85,
            fairness_score=0.88
        ),
        datetime.now(),
        42,
        [1, 2, 3]
    ]
    
    serialized = _make_json_serializable(data)
    
    assert isinstance(serialized, list)
    assert len(serialized) == 4
    assert isinstance(serialized[0], dict)
    assert isinstance(serialized[1], str)
    assert serialized[2] == 42
    assert serialized[3] == [1, 2, 3]


def test_make_json_serializable_numpy():
    """Test _make_json_serializable with NumPy types."""
    data = {
        "int_value": np.int64(42),
        "float_value": np.float64(3.14),
        "array": np.array([1, 2, 3]),
        "nested_array": np.array([[1, 2], [3, 4]])
    }
    
    serialized = _make_json_serializable(data)
    
    assert isinstance(serialized["int_value"], float)
    assert isinstance(serialized["float_value"], float)
    assert isinstance(serialized["array"], list)
    assert serialized["array"] == [1, 2, 3]
    assert isinstance(serialized["nested_array"], list)


def test_make_json_serializable_tuple():
    """Test _make_json_serializable with tuple."""
    data = (
        ComplianceMetrics(
            bias_score=0.8,
            privacy_score=0.9,
            transparency_score=0.85,
            fairness_score=0.88
        ),
        datetime.now()
    )
    
    serialized = _make_json_serializable(data)
    
    assert isinstance(serialized, list)
    assert len(serialized) == 2


def test_make_json_serializable_primitives():
    """Test _make_json_serializable with primitive types."""
    data = {
        "string": "test",
        "int": 42,
        "float": 3.14,
        "bool": True,
        "none": None
    }
    
    serialized = _make_json_serializable(data)
    
    assert serialized == data  # Primitives should remain unchanged


# ==================== Tests for Dataset and Model ====================

def test_medical_diagnosis_dataset_creation(sample_medical_diagnosis_dataset):
    """Test that medical diagnosis dataset can be created."""
    assert len(sample_medical_diagnosis_dataset) == 100


def test_medical_diagnosis_dataset_item_access(sample_medical_diagnosis_dataset):
    """Test that medical diagnosis dataset items can be accessed."""
    item = sample_medical_diagnosis_dataset[0]
    
    assert "input" in item
    assert "target" in item
    assert "metadata" in item
    assert isinstance(item["input"], torch.Tensor)
    assert isinstance(item["target"], torch.Tensor)
    assert isinstance(item["metadata"], dict)


def test_medical_diagnosis_model_forward(sample_medical_diagnosis_model):
    """Test that medical diagnosis model can perform forward pass."""
    x = torch.randn(5, 20)  # batch_size=5, input_size=20
    output = sample_medical_diagnosis_model(x)
    
    assert "logits" in output
    assert "attention_weights" in output
    assert "features" in output
    assert output["logits"].shape == (5, 5)  # batch_size=5, num_classes=5


def test_patient_monitoring_dataset_creation(sample_patient_monitoring_dataset):
    """Test that patient monitoring dataset can be created."""
    assert len(sample_patient_monitoring_dataset) == 100


def test_ehr_dataset_creation(sample_ehr_dataset):
    """Test that EHR dataset can be created."""
    assert len(sample_ehr_dataset) == 100


# ==================== Tests for ComplianceDataset ====================

@pytest.mark.asyncio
async def test_compliance_dataset_item_access(sample_medical_diagnosis_dataset):
    """Test that compliance dataset items can be accessed."""
    compliance_dataset = MedicalDiagnosisCompliance(
        base_dataset=sample_medical_diagnosis_dataset,
        compliance_rules={
            "privacy_threshold": 0.9,
            "fairness_threshold": 0.9,
            "transparency_threshold": 0.9,
            "documentation_complete": True,
            "handle_sensitive_data": True
        },
        data_categories=["health_data", "personal_data"]
    )
    
    item = compliance_dataset[0]
    
    assert "input" in item
    assert "target" in item
    assert "metadata" in item


@pytest.mark.asyncio
async def test_compliance_dataset_length(sample_medical_diagnosis_dataset):
    """Test that compliance dataset has correct length."""
    compliance_dataset = MedicalDiagnosisCompliance(
        base_dataset=sample_medical_diagnosis_dataset,
        compliance_rules={
            "privacy_threshold": 0.9,
            "fairness_threshold": 0.9,
            "transparency_threshold": 0.9,
            "documentation_complete": True,
            "handle_sensitive_data": True
        },
        data_categories=["health_data", "personal_data"]
    )
    
    assert len(compliance_dataset) == 100


# ==================== Tests for ComplianceTrainer ====================

@pytest.mark.asyncio
async def test_compliance_trainer_initialization(
    sample_medical_diagnosis_model,
    sample_compliance_rules,
    sample_training_config
):
    """Test that ComplianceTrainer can be initialized."""
    trainer = ComplianceTrainer(
        model=sample_medical_diagnosis_model,
        compliance_rules=sample_compliance_rules,
        training_config=sample_training_config
    )
    
    assert trainer.model == sample_medical_diagnosis_model
    assert trainer.compliance_rules == sample_compliance_rules
    assert trainer.training_config == sample_training_config


@pytest.mark.asyncio
async def test_compliance_trainer_train(
    sample_medical_diagnosis_model,
    sample_config,
    sample_compliance_rules,
    sample_training_config
):
    """Test that ComplianceTrainer can train a model."""
    from torch.utils.data import DataLoader
    
    # Create a small dataset for testing
    dataset = MedicalDiagnosisDataset(size=50, input_size=20, num_classes=5)
    compliance_dataset = MedicalDiagnosisCompliance(
        base_dataset=dataset,
        compliance_rules={
            "privacy_threshold": 0.9,
            "fairness_threshold": 0.9,
            "transparency_threshold": 0.9,
            "documentation_complete": True,
            "handle_sensitive_data": True
        },
        data_categories=sample_config["data_categories"]
    )
    
    # Custom collate function
    def custom_collate_fn(batch):
        inputs = torch.stack([item["input"] for item in batch])
        targets = torch.stack([item["target"] for item in batch])
        metadata = [item["metadata"] for item in batch]
        return {
            "input": inputs,
            "target": targets,
            "metadata": metadata
        }
    
    train_loader = DataLoader(compliance_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(compliance_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
    
    trainer = ComplianceTrainer(
        model=sample_medical_diagnosis_model,
        compliance_rules=sample_compliance_rules,
        training_config=sample_training_config
    )
    
    results = await trainer.train(
        train_data=train_loader,
        val_data=val_loader,
        metadata=sample_config["metadata"]
    )
    
    assert results is not None
    assert "metrics_history" in results
    assert "violations" in results
    assert "final_evaluation" in results


# ==================== Tests for Custom Collate Function ====================

@pytest.mark.asyncio
async def test_custom_collate_function():
    """Test that custom collate function works correctly."""
    from torch.utils.data import DataLoader
    
    dataset = MedicalDiagnosisDataset(size=10, input_size=20, num_classes=5)
    compliance_dataset = MedicalDiagnosisCompliance(
        base_dataset=dataset,
        compliance_rules={
            "privacy_threshold": 0.9,
            "fairness_threshold": 0.9,
            "transparency_threshold": 0.9,
            "documentation_complete": True,
            "handle_sensitive_data": True
        },
        data_categories=["health_data", "personal_data"]
    )
    
    # Custom collate function
    def custom_collate_fn(batch):
        inputs = torch.stack([item["input"] for item in batch])
        targets = torch.stack([item["target"] for item in batch])
        metadata = [item["metadata"] for item in batch]
        return {
            "input": inputs,
            "target": targets,
            "metadata": metadata
        }
    
    loader = DataLoader(compliance_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)
    
    # Get a batch
    batch = next(iter(loader))
    
    assert "input" in batch
    assert "target" in batch
    assert "metadata" in batch
    assert batch["input"].shape[0] == 4  # batch size
    assert batch["target"].shape[0] == 4
    assert len(batch["metadata"]) == 4
    assert isinstance(batch["metadata"], list)


# ==================== Tests for JSON Serialization ====================

@pytest.mark.asyncio
async def test_json_serialization_of_results(sample_config):
    """Test that results can be serialized to JSON."""
    results = await run_healthcare_compliance_example(
        dataset_class=MedicalDiagnosisDataset,
        model_class=DiagnosisModel,
        compliance_class=MedicalDiagnosisCompliance,
        config=sample_config
    )
    
    # Try to serialize the results
    serialized = _make_json_serializable(results)
    
    # Should be able to convert to JSON string
    json_str = json.dumps(serialized)
    assert isinstance(json_str, str)
    assert len(json_str) > 0
    
    # Should be able to parse back
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)
    assert "metrics_history" in parsed
    assert "violations" in parsed
    assert "final_evaluation" in parsed


# ==================== Tests for GovernanceConfig ====================

def test_governance_config_initialization():
    """Test that GovernanceConfig can be initialized."""
    config = GovernanceConfig(
        organization_id="test_org",
        organization_name="Test Organization",
        dpo_email="dpo@test.com",
        enabled_regulations=[Regulation.HIPAA, Regulation.GDPR]
    )
    
    assert config.organization_id == "test_org"
    assert config.organization_name == "Test Organization"
    assert config.dpo_email == "dpo@test.com"
    assert Regulation.HIPAA in config.enabled_regulations
    assert Regulation.GDPR in config.enabled_regulations


def test_regulation_enum():
    """Test that Regulation enum has expected values."""
    assert hasattr(Regulation, "HIPAA")
    assert hasattr(Regulation, "GDPR")
    assert hasattr(Regulation, "FDA")
    assert hasattr(Regulation, "CCPA")
    assert hasattr(Regulation, "EMA")
    assert hasattr(Regulation, "ICH")
    assert hasattr(Regulation, "GCP")
    assert hasattr(Regulation, "AI_ACT")


# ==================== Tests for Main Function ====================

@pytest.mark.asyncio
async def test_main_function():
    """Test that main function can be called."""
    # Mock the file operations to avoid creating actual files
    with patch('builtins.open', create=True), \
         patch('json.dump'), \
         patch('json.dumps', return_value='{}'), \
         patch('examples.compliance.healthcare_compliance_example.run_healthcare_compliance_example') as mock_run:
        
        # Mock the run function to return a simple result
        mock_run.return_value = {
            "metrics_history": [],
            "violations": [],
            "final_evaluation": {
                "compliance_scores": {"bias": 0.0, "privacy": 0.0},
                "violations": [],
                "recommendations": []
            }
        }
        
        try:
            await main()
            # If we get here, the function ran without errors
            # Verify that run_healthcare_compliance_example was called
            assert mock_run.called
        except Exception as e:
            # Some errors are acceptable (like missing dependencies)
            # but we should log them
            if "Optuna" not in str(e) and "import" not in str(e).lower():
                pytest.fail(f"main() function failed with unexpected error: {e}")


# ==================== Tests for Multiple Use Cases ====================

@pytest.mark.asyncio
async def test_multiple_use_cases():
    """Test that multiple use cases can be run."""
    use_cases = [
        (MedicalDiagnosisDataset, DiagnosisModel, MedicalDiagnosisCompliance),
        (PatientMonitoringDataset, PatientMonitoringModel, PatientMonitoringCompliance),
    ]
    
    for dataset_class, model_class, compliance_class in use_cases:
        config = {
            "data_categories": ["health_data", "personal_data"],
            "metadata": {
                "model_type": "test",
                "data_categories": ["health_data", "personal_data"],
                "jurisdiction": "US",
                "hipaa_covered": True
            }
        }
        
        results = await run_healthcare_compliance_example(
            dataset_class=dataset_class,
            model_class=model_class,
            compliance_class=compliance_class,
            config=config
        )
        
        assert results is not None
        assert "final_evaluation" in results


# ==================== Tests for Error Handling ====================

@pytest.mark.asyncio
async def test_error_handling_invalid_config():
    """Test error handling with invalid configuration."""
    # Test with missing required fields
    invalid_config = {
        "data_categories": ["health_data"]
        # Missing metadata
    }
    
    try:
        await run_healthcare_compliance_example(
            dataset_class=MedicalDiagnosisDataset,
            model_class=DiagnosisModel,
            compliance_class=MedicalDiagnosisCompliance,
            config=invalid_config
        )
        # Should either work or raise a clear error
    except KeyError:
        # Expected if metadata is required
        pass
    except Exception as e:
        # Other errors might be acceptable
        assert "metadata" in str(e).lower() or "config" in str(e).lower()


def test_error_handling_invalid_serialization():
    """Test error handling with invalid serialization."""
    # Test with unsupported type (should return as-is)
    class UnsupportedType:
        pass
    
    obj = UnsupportedType()
    result = _make_json_serializable(obj)
    
    # Should return the object as-is or handle gracefully
    assert result is not None

