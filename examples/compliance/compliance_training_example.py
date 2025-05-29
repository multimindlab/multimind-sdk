"""
Example script demonstrating general compliance monitoring and evaluation.
This script provides a template for implementing compliance monitoring in various domains.
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

class ExampleDataset(Dataset):
    """Example dataset for compliance monitoring."""
    
    def __init__(self, size: int, input_size: int, num_classes: int):
        self.size = size
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Generate synthetic data
        self.data = torch.randn(size, input_size)
        self.labels = torch.randint(0, num_classes, (size,))
        
        # Add metadata for compliance checks
        self.metadata = {
            "data_categories": ["personal_data", "sensitive_data"],
            "jurisdiction": "US",
            "regulations": ["GDPR", "CCPA"],
            "consent_status": True,
            "data_retention_period": 365,  # days
            "data_minimization": True,
            "purpose_limitation": True,
            "transparency": True
        }
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "data": self.data[idx],
            "label": self.labels[idx],
            "metadata": self.metadata
        }

class ExampleModel(nn.Module):
    """Example model with explainability features."""
    
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Attention mechanism for explainability
        self.attention = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, num_classes)
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
        
        # Classify
        logits = self.classifier(attended_features)
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "features": features
        }

class ExampleCompliance(ComplianceDataset):
    """Example compliance dataset wrapper."""
    
    def __init__(
        self,
        base_dataset: Dataset,
        compliance_rules: Dict[str, Any],
        data_categories: List[str]
    ):
        super().__init__(base_dataset, compliance_rules, data_categories)
    
    async def check_privacy_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check privacy compliance."""
        return {
            "data_minimization": data["metadata"]["data_minimization"],
            "purpose_limitation": data["metadata"]["purpose_limitation"],
            "consent_status": data["metadata"]["consent_status"],
            "data_retention": data["metadata"]["data_retention_period"] <= 365
        }
    
    async def check_fairness_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check fairness compliance."""
        return {
            "demographic_parity": True,
            "equal_opportunity": True,
            "disparate_impact": True
        }
    
    async def check_transparency_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check transparency compliance."""
        return {
            "explainability": True,
            "documentation": True,
            "audit_trail": True
        }

async def main():
    # Initialize governance config
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
    
    # Create model and datasets
    model = ExampleModel(input_size=20, num_classes=5)
    base_dataset = ExampleDataset(size=1000, input_size=20, num_classes=5)
    
    # Wrap dataset with compliance checks
    compliance_dataset = ExampleCompliance(
        base_dataset=base_dataset,
        compliance_rules={
            "privacy_threshold": 0.9,
            "fairness_threshold": 0.9,
            "transparency_threshold": 0.9,
            "documentation_complete": True
        },
        data_categories=["personal_data", "sensitive_data"]
    )
    
    # Create data loaders
    train_loader = DataLoader(compliance_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(compliance_dataset, batch_size=32, shuffle=False)
    
    # Configure compliance training
    compliance_rules = {
        "bias_threshold": 0.1,
        "privacy_threshold": 0.9,
        "transparency_threshold": 0.9,
        "fairness_threshold": 0.9,
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
            "fairness"
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
            "model_type": "example",
            "data_categories": ["personal_data", "sensitive_data"],
            "jurisdiction": "US",
            "sensitive_data": True,
            "explainability_required": True
        }
    )
    
    # Save results
    results_path = "compliance_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print compliance evaluation results
    print("\nCompliance Evaluation Results:")
    print(json.dumps(results["final_evaluation"], indent=2))
    
    # Print recommendations
    print("\nRecommendations:")
    for rec in results["final_evaluation"]["recommendations"]:
        print(f"- {rec['action']} (Priority: {rec['priority']})")

if __name__ == "__main__":
    asyncio.run(main()) 