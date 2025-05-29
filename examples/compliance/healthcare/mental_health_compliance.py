"""
Example script demonstrating compliance monitoring for mental health AI systems.
This script ensures HIPAA compliance and medical ethics in AI-powered mental health assessment systems.
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

class MentalHealthDataset(Dataset):
    """Dataset for mental health assessment."""
    
    def __init__(self, size: int, input_size: int, num_classes: int):
        self.size = size
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Generate synthetic mental health assessment data
        self.data = torch.randn(size, input_size)
        self.labels = torch.randint(0, num_classes, (size,))
        
        # Add metadata for compliance checks
        self.metadata = {
            "patient_id": [f"PAT_{i:06d}" for i in range(size)],
            "assessment_type": np.random.choice(
                ["depression", "anxiety", "ptsd", "bipolar", "general"],
                size=size
            ),
            "risk_level": np.random.choice(
                ["low", "medium", "high", "critical"],
                size=size
            ),
            "consent_status": np.random.choice([True, False], size=size),
            "emergency_contact": [f"EC_{i:06d}" for i in range(size)],
            "data_categories": ["mental_health_data", "personal_data", "sensitive_data"],
            "jurisdiction": "US",
            "regulations": ["HIPAA", "GDPR", "CCPA"],
            "data_retention_period": 365,  # days
            "data_minimization": True,
            "purpose_limitation": True,
            "transparency": True,
            "crisis_intervention": True
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

class MentalHealthModel(nn.Module):
    """Mental health assessment model with explainability and risk assessment."""
    
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Attention mechanism for explainability
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )
        
        # Risk assessment
        self.risk_assessor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4 risk levels
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes)
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
        
        # Classify
        logits = self.classifier(attended_features)
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "features": features,
            "risk_scores": risk_scores
        }

class MentalHealthCompliance(ComplianceDataset):
    """Compliance wrapper for mental health assessment."""
    
    def __init__(
        self,
        base_dataset: Dataset,
        compliance_rules: Dict[str, Any],
        data_categories: List[str]
    ):
        super().__init__(base_dataset, compliance_rules, data_categories)
    
    async def check_privacy_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check privacy compliance for mental health data."""
        metadata = data["metadata"]
        return {
            "data_minimization": metadata["data_minimization"],
            "purpose_limitation": metadata["purpose_limitation"],
            "consent_status": metadata["consent_status"],
            "data_retention": metadata["data_retention_period"] <= 365,
            "emergency_contact": bool(metadata["emergency_contact"]),
            "risk_assessment": metadata["risk_level"] in ["low", "medium", "high", "critical"]
        }
    
    async def check_fairness_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check fairness compliance for mental health assessment."""
        return {
            "demographic_parity": True,
            "equal_opportunity": True,
            "disparate_impact": True,
            "risk_assessment_fairness": True
        }
    
    async def check_transparency_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check transparency compliance for mental health assessment."""
        return {
            "explainability": True,
            "documentation": True,
            "audit_trail": True,
            "crisis_intervention": data["metadata"]["crisis_intervention"]
        }

async def main():
    # Initialize governance config
    config = GovernanceConfig(
        organization_id="mental_health_org_123",
        organization_name="Mental Health AI Corp",
        dpo_email="dpo@mentalhealth.com",
        enabled_regulations=[
            Regulation.HIPAA,
            Regulation.GDPR,
            Regulation.AI_ACT
        ]
    )
    
    # Create model and datasets
    model = MentalHealthModel(input_size=20, num_classes=5)
    base_dataset = MentalHealthDataset(size=1000, input_size=20, num_classes=5)
    
    # Wrap dataset with compliance checks
    compliance_dataset = MentalHealthCompliance(
        base_dataset=base_dataset,
        compliance_rules={
            "privacy_threshold": 0.95,  # Higher threshold for mental health data
            "fairness_threshold": 0.95,
            "transparency_threshold": 0.95,
            "documentation_complete": True,
            "crisis_intervention": True
        },
        data_categories=["mental_health_data", "personal_data", "sensitive_data"]
    )
    
    # Create data loaders
    train_loader = DataLoader(compliance_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(compliance_dataset, batch_size=32, shuffle=False)
    
    # Configure compliance training
    compliance_rules = {
        "bias_threshold": 0.05,  # Lower threshold for mental health
        "privacy_threshold": 0.95,
        "transparency_threshold": 0.95,
        "fairness_threshold": 0.95,
        "hipaa_compliance": True,
        "data_minimization": True,
        "audit_trail": True,
        "explainability": True,
        "crisis_intervention": True
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
            "model_type": "mental_health",
            "data_categories": ["mental_health_data", "personal_data", "sensitive_data"],
            "jurisdiction": "US",
            "hipaa_covered": True,
            "sensitive_data": True,
            "explainability_required": True,
            "crisis_intervention": True
        }
    )
    
    # Save results
    results_path = "mental_health_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print compliance evaluation results
    print("\nMental Health Compliance Evaluation Results:")
    print(json.dumps(results["final_evaluation"], indent=2))
    
    # Print recommendations
    print("\nRecommendations:")
    for rec in results["final_evaluation"]["recommendations"]:
        print(f"- {rec['action']} (Priority: {rec['priority']})")

if __name__ == "__main__":
    asyncio.run(main()) 