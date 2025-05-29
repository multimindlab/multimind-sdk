"""
Example script demonstrating compliance monitoring for healthcare fraud detection systems.
This script ensures regulatory compliance and ethical standards in AI-powered healthcare fraud detection.
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

class FraudDetectionDataset(Dataset):
    """Dataset for healthcare fraud detection."""
    
    def __init__(self, size: int, input_size: int, num_classes: int):
        self.size = size
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Generate synthetic fraud detection data
        self.data = torch.randn(size, input_size)
        self.labels = torch.randint(0, num_classes, (size,))
        
        # Add metadata for compliance checks
        self.metadata = {
            "claim_id": [f"CLM_{i:06d}" for i in range(size)],
            "claim_type": np.random.choice(
                ["medical", "dental", "pharmacy", "hospital", "outpatient"],
                size=size
            ),
            "provider_id": [f"PRV_{i:04d}" for i in range(size)],
            "patient_id": [f"PAT_{i:06d}" for i in range(size)],
            "claim_amount": np.random.uniform(100, 10000, size=size),
            "risk_score": np.random.choice(
                ["low", "medium", "high", "critical"],
                size=size
            ),
            "data_categories": ["claims_data", "personal_data", "financial_data"],
            "jurisdiction": "US",
            "regulations": ["HIPAA", "GDPR", "CCPA", "PCI-DSS"],
            "data_retention_period": 365,  # days
            "data_minimization": True,
            "purpose_limitation": True,
            "transparency": True,
            "fraud_monitoring": True
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

class FraudDetectionModel(nn.Module):
    """Fraud detection model with explainability and risk assessment."""
    
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
        
        # Risk assessment
        self.risk_assessor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 risk levels
        )
        
        # Claim type classifier
        self.claim_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5 claim types
        )
        
        # Fraud detector
        self.fraud_detector = nn.Sequential(
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
        
        # Assess risk
        risk_scores = self.risk_assessor(attended_features)
        
        # Classify claim type
        claim_scores = self.claim_classifier(attended_features)
        
        # Detect fraud
        fraud_scores = self.fraud_detector(attended_features)
        
        return {
            "logits": fraud_scores,
            "attention_weights": attention_weights,
            "features": features,
            "risk_scores": risk_scores,
            "claim_scores": claim_scores
        }

class FraudDetectionCompliance(ComplianceDataset):
    """Compliance wrapper for healthcare fraud detection."""
    
    def __init__(
        self,
        base_dataset: Dataset,
        compliance_rules: Dict[str, Any],
        data_categories: List[str]
    ):
        super().__init__(base_dataset, compliance_rules, data_categories)
    
    async def check_privacy_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check privacy compliance for fraud detection data."""
        metadata = data["metadata"]
        return {
            "data_minimization": metadata["data_minimization"],
            "purpose_limitation": metadata["purpose_limitation"],
            "fraud_monitoring": metadata["fraud_monitoring"],
            "data_retention": metadata["data_retention_period"] <= 365,
            "provider_verification": bool(metadata["provider_id"]),
            "patient_verification": bool(metadata["patient_id"])
        }
    
    async def check_fairness_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check fairness compliance for fraud detection."""
        return {
            "demographic_parity": True,
            "equal_opportunity": True,
            "disparate_impact": True,
            "risk_assessment_fairness": True
        }
    
    async def check_transparency_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check transparency compliance for fraud detection."""
        return {
            "explainability": True,
            "documentation": True,
            "audit_trail": True,
            "risk_assessment": data["metadata"]["risk_score"] in ["low", "medium", "high", "critical"]
        }

async def main():
    # Initialize governance config
    config = GovernanceConfig(
        organization_id="fraud_detection_org_123",
        organization_name="Healthcare Fraud Detection AI Corp",
        dpo_email="dpo@frauddetection.com",
        enabled_regulations=[
            Regulation.HIPAA,
            Regulation.GDPR,
            Regulation.CCPA,
            Regulation.PCI_DSS
        ]
    )
    
    # Create model and datasets
    model = FraudDetectionModel(input_size=20, num_classes=5)
    base_dataset = FraudDetectionDataset(size=1000, input_size=20, num_classes=5)
    
    # Wrap dataset with compliance checks
    compliance_dataset = FraudDetectionCompliance(
        base_dataset=base_dataset,
        compliance_rules={
            "privacy_threshold": 0.95,
            "fairness_threshold": 0.95,
            "transparency_threshold": 0.95,
            "documentation_complete": True,
            "fraud_monitoring": True
        },
        data_categories=["claims_data", "personal_data", "financial_data"]
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
        "fraud_monitoring": True
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
            "risk_assessment"
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
            "model_type": "fraud_detection",
            "data_categories": ["claims_data", "personal_data", "financial_data"],
            "jurisdiction": "US",
            "hipaa_covered": True,
            "sensitive_data": True,
            "explainability_required": True,
            "fraud_monitoring": True
        }
    )
    
    # Save results
    results_path = "fraud_detection_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print compliance evaluation results
    print("\nFraud Detection Compliance Evaluation Results:")
    print(json.dumps(results["final_evaluation"], indent=2))
    
    # Print recommendations
    print("\nRecommendations:")
    for rec in results["final_evaluation"]["recommendations"]:
        print(f"- {rec['action']} (Priority: {rec['priority']})")

if __name__ == "__main__":
    asyncio.run(main()) 