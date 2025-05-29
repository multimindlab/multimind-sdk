"""
Example script demonstrating compliance monitoring for medical imaging analysis systems.
This script ensures HIPAA compliance and medical ethics in AI-powered medical imaging analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from multimind.compliance.model_training import (
    ComplianceDataset,
    ComplianceTrainer,
    ComplianceMetrics
)
from multimind.compliance import GovernanceConfig, Regulation
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

class MedicalImagingDataset(Dataset):
    """Dataset for medical imaging analysis."""
    
    def __init__(self, size: int, input_size: int, num_classes: int):
        self.size = size
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Generate synthetic medical imaging data
        self.data = torch.randn(size, input_size)
        self.labels = torch.randint(0, num_classes, (size,))
        
        # Add metadata for compliance checks
        self.metadata = {
            "patient_id": [f"PAT_{i:06d}" for i in range(size)],
            "image_type": np.random.choice(
                ["xray", "mri", "ct", "ultrasound", "pet"],
                size=size
            ),
            "body_region": np.random.choice(
                ["brain", "chest", "abdomen", "spine", "limbs"],
                size=size
            ),
            "image_quality": np.random.choice(
                ["low", "medium", "high"],
                size=size
            ),
            "radiologist_id": [f"RAD_{i:04d}" for i in range(size)],
            "data_categories": ["medical_imaging", "personal_data", "sensitive_data"],
            "jurisdiction": "US",
            "regulations": ["HIPAA", "GDPR", "CCPA"],
            "data_retention_period": 365,  # days
            "data_minimization": True,
            "purpose_limitation": True,
            "transparency": True,
            "image_anonymization": True
        }
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "data": self.data[idx],
            "label": self.labels[idx],
            "metadata": {
                k: v[idx] if isinstance(v, (list, np.ndarray)) else v
                for k, v in self.metadata.items()
            }
        }

class MedicalImagingModel(nn.Module):
    """Medical imaging analysis model with explainability and quality assessment."""
    
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Attention mechanism for explainability
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # Quality assessment
        self.quality_assessor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 quality levels
        )
        
        # Region detection
        self.region_detector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5 body regions
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes)
        )
        
        # Compliance monitoring
        self.compliance_metrics = ComplianceMetrics()
    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Calculate attention weights
        attention_weights = self.attention(features)
        
        # Apply attention
        attended_features = features * attention_weights
        
        # Assess quality
        quality_scores = self.quality_assessor(attended_features)
        
        # Detect regions
        region_scores = self.region_detector(attended_features)
        
        # Classify
        logits = self.classifier(attended_features)
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "features": features,
            "quality_scores": quality_scores,
            "region_scores": region_scores
        }

class MedicalImagingCompliance(ComplianceDataset):
    """Compliance wrapper for medical imaging analysis."""
    
    def __init__(
        self,
        base_dataset: Dataset,
        compliance_rules: Dict[str, Any],
        data_categories: List[str]
    ):
        super().__init__(base_dataset, compliance_rules, data_categories)
    
    async def check_privacy_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check privacy compliance for medical imaging data."""
        metadata = data["metadata"]
        return {
            "data_minimization": metadata["data_minimization"],
            "purpose_limitation": metadata["purpose_limitation"],
            "image_anonymization": metadata["image_anonymization"],
            "data_retention": metadata["data_retention_period"] <= 365,
            "radiologist_verification": bool(metadata["radiologist_id"])
        }
    
    async def check_fairness_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check fairness compliance for medical imaging analysis."""
        return {
            "demographic_parity": True,
            "equal_opportunity": True,
            "disparate_impact": True,
            "quality_assessment_fairness": True
        }
    
    async def check_transparency_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check transparency compliance for medical imaging analysis."""
        return {
            "explainability": True,
            "documentation": True,
            "audit_trail": True,
            "quality_assessment": data["metadata"]["image_quality"] in ["low", "medium", "high"]
        }

async def main():
    # Initialize governance config
    config = GovernanceConfig(
        organization_id="imaging_org_123",
        organization_name="Medical Imaging AI Corp",
        dpo_email="dpo@imaging.com",
        enabled_regulations=[
            Regulation.HIPAA,
            Regulation.GDPR,
            Regulation.AI_ACT
        ]
    )
    
    # Create model and datasets
    model = MedicalImagingModel(input_size=20, num_classes=5)
    base_dataset = MedicalImagingDataset(size=1000, input_size=20, num_classes=5)
    
    # Wrap dataset with compliance checks
    compliance_dataset = MedicalImagingCompliance(
        base_dataset=base_dataset,
        compliance_rules={
            "privacy_threshold": 0.95,
            "fairness_threshold": 0.95,
            "transparency_threshold": 0.95,
            "documentation_complete": True,
            "image_anonymization": True
        },
        data_categories=["medical_imaging", "personal_data", "sensitive_data"]
    )
    
    # Create data loaders
    train_loader = DataLoader(compliance_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(compliance_dataset, batch_size=32, shuffle=False)
    
    # Configure compliance training
    compliance_rules = {
        "bias_threshold": 0.05,
        "privacy_threshold": 0.95,
        "transparency_threshold": 0.95,
        "fairness_threshold": 0.95,
        "hipaa_compliance": True,
        "data_minimization": True,
        "audit_trail": True,
        "explainability": True,
        "image_anonymization": True
    }
    
    training_config = {
        "epochs": 10,
        "thresholds": compliance_rules,
        "evaluation_metrics": [
            "bias",
            "privacy",
            "transparency",
            "fairness",
            "hipaa_compliance",
            "quality_assessment"
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
        metadata={
            "model_type": "medical_imaging",
            "data_categories": ["medical_imaging", "personal_data", "sensitive_data"],
            "jurisdiction": "US",
            "hipaa_covered": True,
            "sensitive_data": True,
            "explainability_required": True,
            "image_anonymization": True
        }
    )
    
    # Save results
    results_path = "medical_imaging_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print compliance evaluation results
    print("\nMedical Imaging Compliance Evaluation Results:")
    print(json.dumps(results["final_evaluation"], indent=2))
    
    # Print recommendations
    print("\nRecommendations:")
    for rec in results["final_evaluation"]["recommendations"]:
        print(f"- {rec['action']} (Priority: {rec['priority']})")

if __name__ == "__main__":
    asyncio.run(main()) 