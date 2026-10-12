"""
Example script demonstrating healthcare compliance monitoring and evaluation.
This script provides examples for various healthcare use cases including:
- Medical Diagnosis
- Patient Monitoring
- Medical Imaging
- Clinical Trials
- Electronic Health Records (EHR)
- Medical Devices
- Medical Research
- Telemedicine
- Mental Health
- Medical Imaging Analysis
- Drug Discovery
- Fraud Detection
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from examples.compliance.healthcare.clinical_trial_compliance import (
    ClinicalTrialCompliance,
    ClinicalTrialDataset,
    ClinicalTrialModel,
)
from examples.compliance.healthcare.drug_discovery_compliance import (
    DrugDiscoveryCompliance,
    DrugDiscoveryDataset,
    DrugDiscoveryModel,
)
from examples.compliance.healthcare.ehr_compliance import EHRCompliance, EHRDataset, EHRModel
from examples.compliance.healthcare.fraud_detection_compliance import (
    FraudDetectionCompliance,
    FraudDetectionDataset,
    FraudDetectionModel,
)
from examples.compliance.healthcare.medical_device_compliance import (
    MedicalDeviceCompliance,
    MedicalDeviceDataset,
    MedicalDeviceModel,
)

# Import healthcare-specific examples
from examples.compliance.healthcare.medical_diagnosis_compliance import (
    DiagnosisModel,
    MedicalDiagnosisCompliance,
    MedicalDiagnosisDataset,
)
from examples.compliance.healthcare.medical_imaging_analysis_compliance import (
    MedicalImagingCompliance,
    MedicalImagingDataset,
    MedicalImagingModel,
)
from examples.compliance.healthcare.medical_imaging_compliance import (
    MedicalImagingCompliance,
    MedicalImagingDataset,
    MedicalImagingModel,
)
from examples.compliance.healthcare.medical_research_compliance import (
    MedicalResearchCompliance,
    MedicalResearchDataset,
    MedicalResearchModel,
)
from examples.compliance.healthcare.mental_health_compliance import (
    MentalHealthCompliance,
    MentalHealthDataset,
    MentalHealthModel,
)
from examples.compliance.healthcare.patient_monitoring_compliance import (
    PatientMonitoringCompliance,
    PatientMonitoringDataset,
    PatientMonitoringModel,
)
from examples.compliance.healthcare.telemedicine_compliance import (
    TelemedicineCompliance,
    TelemedicineDataset,
    TelemedicineModel,
)
from multimind.compliance import GovernanceConfig, Regulation
from multimind.compliance.model_training import (
    ComplianceDataset,
    ComplianceMetrics,
    ComplianceTrainer,
)


def _make_json_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, ComplianceMetrics):
        return {
            "bias_score": obj.bias_score,
            "privacy_score": obj.privacy_score,
            "transparency_score": obj.transparency_score,
            "fairness_score": obj.fairness_score,
            "timestamp": obj.timestamp.isoformat() if isinstance(obj.timestamp, datetime) else str(obj.timestamp)
        }
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

async def run_healthcare_compliance_example(
    dataset_class,
    model_class,
    compliance_class,
    config: Dict[str, Any]
):
    """Run compliance monitoring for a specific healthcare use case."""

    # Create model and datasets
    model = model_class(input_size=20, num_classes=5)
    base_dataset = dataset_class(size=1000, input_size=20, num_classes=5)

    # Wrap dataset with compliance checks
    compliance_dataset = compliance_class(
        base_dataset=base_dataset,
        compliance_rules={
            "privacy_threshold": 0.9,
            "fairness_threshold": 0.9,
            "transparency_threshold": 0.9,
            "documentation_complete": True,
            "handle_sensitive_data": True  # Required for MedicalDiagnosisCompliance
        },
        data_categories=config["data_categories"]
    )

    # Custom collate function to handle variable-length metadata
    def custom_collate_fn(batch):
        """Custom collate function to handle variable-length metadata."""
        inputs = torch.stack([item["input"] for item in batch])
        targets = torch.stack([item["target"] for item in batch])
        metadata = [item["metadata"] for item in batch]  # Keep as list of dicts
        return {
            "input": inputs,
            "target": targets,
            "metadata": metadata
        }

    # Create data loaders
    train_loader = DataLoader(compliance_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(compliance_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    # Configure compliance training
    compliance_rules = {
        "bias_threshold": 0.1,
        "privacy_threshold": 0.9,
        "transparency_threshold": 0.9,
        "fairness_threshold": 0.9,
        "hipaa_compliance": True,
        "data_minimization": True,
        "audit_trail": True,
        "explainability": True
    }

    training_config = {
        "epochs": 10,
        "thresholds": compliance_rules,
        "evaluation_metrics": [
            "bias",
            "privacy",
            "transparency",
            "fairness",
            "hipaa_compliance"
        ]
    }

    # Initialize compliance trainer
    trainer = ComplianceTrainer(
        model=model,
        compliance_rules=compliance_rules,
        training_config=training_config
    )

    # Train model with compliance monitoring
    results = await trainer.train(
        train_data=train_loader,
        val_data=val_loader,
        metadata=config["metadata"]
    )

    return results

async def main():
    # Initialize governance config
    config = GovernanceConfig(
        organization_id="health_org_123",
        organization_name="Healthcare AI Corp",
        dpo_email="dpo@healthcare.com",
        enabled_regulations=[
            Regulation.HIPAA,
            Regulation.GDPR,
            Regulation.AI_ACT
        ]
    )

    # Define configurations for each healthcare use case
    use_cases = {
        "medical_diagnosis": {
            "dataset_class": MedicalDiagnosisDataset,
            "model_class": DiagnosisModel,
            "compliance_class": MedicalDiagnosisCompliance,
            "data_categories": ["health_data", "personal_data"],
            "metadata": {
                "model_type": "medical_diagnosis",
                "data_categories": ["health_data", "personal_data"],
                "jurisdiction": "US",
                "hipaa_covered": True,
                "sensitive_data": True,
                "explainability_required": True
            }
        },
        "patient_monitoring": {
            "dataset_class": PatientMonitoringDataset,
            "model_class": PatientMonitoringModel,
            "compliance_class": PatientMonitoringCompliance,
            "data_categories": ["health_data", "personal_data"],
            "metadata": {
                "model_type": "patient_monitoring",
                "data_categories": ["health_data", "personal_data"],
                "jurisdiction": "US",
                "hipaa_covered": True,
                "sensitive_data": True,
                "real_time_required": True
            }
        },
        "medical_imaging": {
            "dataset_class": MedicalImagingDataset,
            "model_class": MedicalImagingModel,
            "compliance_class": MedicalImagingCompliance,
            "data_categories": ["health_data", "personal_data"],
            "metadata": {
                "model_type": "medical_imaging",
                "data_categories": ["health_data", "personal_data"],
                "jurisdiction": "US",
                "hipaa_covered": True,
                "sensitive_data": True,
                "explainability_required": True
            }
        },
        "clinical_trial": {
            "dataset_class": ClinicalTrialDataset,
            "model_class": ClinicalTrialModel,
            "compliance_class": ClinicalTrialCompliance,
            "data_categories": ["clinical_trial_data", "personal_data"],
            "metadata": {
                "model_type": "clinical_trial",
                "data_categories": ["clinical_trial_data", "personal_data"],
                "jurisdiction": "US",
                "hipaa_covered": True,
                "sensitive_data": True,
                "explainability_required": True
            }
        },
        "ehr": {
            "dataset_class": EHRDataset,
            "model_class": EHRModel,
            "compliance_class": EHRCompliance,
            "data_categories": ["health_data", "personal_data"],
            "metadata": {
                "model_type": "ehr",
                "data_categories": ["health_data", "personal_data"],
                "jurisdiction": "US",
                "hipaa_covered": True,
                "sensitive_data": True,
                "explainability_required": True
            }
        },
        "medical_device": {
            "dataset_class": MedicalDeviceDataset,
            "model_class": MedicalDeviceModel,
            "compliance_class": MedicalDeviceCompliance,
            "data_categories": ["device_data", "personal_data"],
            "metadata": {
                "model_type": "medical_device",
                "data_categories": ["device_data", "personal_data"],
                "jurisdiction": "US",
                "hipaa_covered": True,
                "sensitive_data": True,
                "explainability_required": True
            }
        },
        "medical_research": {
            "dataset_class": MedicalResearchDataset,
            "model_class": MedicalResearchModel,
            "compliance_class": MedicalResearchCompliance,
            "data_categories": ["research_data", "personal_data"],
            "metadata": {
                "model_type": "medical_research",
                "data_categories": ["research_data", "personal_data"],
                "jurisdiction": "US",
                "hipaa_covered": True,
                "sensitive_data": True,
                "explainability_required": True
            }
        },
        "telemedicine": {
            "dataset_class": TelemedicineDataset,
            "model_class": TelemedicineModel,
            "compliance_class": TelemedicineCompliance,
            "data_categories": ["health_data", "personal_data"],
            "metadata": {
                "model_type": "telemedicine",
                "data_categories": ["health_data", "personal_data"],
                "jurisdiction": "US",
                "hipaa_covered": True,
                "sensitive_data": True,
                "explainability_required": True
            }
        },
        "mental_health": {
            "dataset_class": MentalHealthDataset,
            "model_class": MentalHealthModel,
            "compliance_class": MentalHealthCompliance,
            "data_categories": ["mental_health_data", "personal_data"],
            "metadata": {
                "model_type": "mental_health",
                "data_categories": ["mental_health_data", "personal_data"],
                "jurisdiction": "US",
                "hipaa_covered": True,
                "sensitive_data": True,
                "explainability_required": True,
                "crisis_intervention": True
            }
        },
        "medical_imaging_analysis": {
            "dataset_class": MedicalImagingDataset,
            "model_class": MedicalImagingModel,
            "compliance_class": MedicalImagingCompliance,
            "data_categories": ["medical_imaging", "personal_data"],
            "metadata": {
                "model_type": "medical_imaging_analysis",
                "data_categories": ["medical_imaging", "personal_data"],
                "jurisdiction": "US",
                "hipaa_covered": True,
                "sensitive_data": True,
                "explainability_required": True,
                "image_anonymization": True
            }
        },
        "drug_discovery": {
            "dataset_class": DrugDiscoveryDataset,
            "model_class": DrugDiscoveryModel,
            "compliance_class": DrugDiscoveryCompliance,
            "data_categories": ["drug_development", "research_data"],
            "metadata": {
                "model_type": "drug_discovery",
                "data_categories": ["drug_development", "research_data"],
                "jurisdiction": "US",
                "fda_covered": True,
                "sensitive_data": True,
                "explainability_required": True,
                "safety_monitoring": True
            }
        },
        "fraud_detection": {
            "dataset_class": FraudDetectionDataset,
            "model_class": FraudDetectionModel,
            "compliance_class": FraudDetectionCompliance,
            "data_categories": ["claims_data", "personal_data"],
            "metadata": {
                "model_type": "fraud_detection",
                "data_categories": ["claims_data", "personal_data"],
                "jurisdiction": "US",
                "hipaa_covered": True,
                "sensitive_data": True,
                "explainability_required": True,
                "fraud_monitoring": True
            }
        }
    }

    # Run compliance monitoring for each use case
    results = {}
    for use_case, config in use_cases.items():
        print(f"\nRunning compliance monitoring for {use_case}...")
        results[use_case] = await run_healthcare_compliance_example(
            dataset_class=config["dataset_class"],
            model_class=config["model_class"],
            compliance_class=config["compliance_class"],
            config=config
        )

        # Save results
        results_path = f"{use_case}_results.json"
        with open(results_path, "w") as f:
            json.dump(_make_json_serializable(results[use_case]), f, indent=2)

        # Print compliance evaluation results
        print(f"\n{use_case.title()} Compliance Evaluation Results:")
        print(json.dumps(_make_json_serializable(results[use_case]["final_evaluation"]), indent=2))

        # Print recommendations
        print("\nRecommendations:")
        for rec in results[use_case]["final_evaluation"]["recommendations"]:
            print(f"- {rec['action']} (Priority: {rec['priority']})")

if __name__ == "__main__":
    asyncio.run(main())
