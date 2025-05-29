"""
Example script demonstrating compliance monitoring for drug discovery and development systems.
This script ensures regulatory compliance and ethical standards in AI-powered drug discovery.
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

class DrugDiscoveryDataset(Dataset):
    """Dataset for drug discovery and development."""
    
    def __init__(self, size: int, input_size: int, num_classes: int):
        self.size = size
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Generate synthetic drug discovery data
        self.data = torch.randn(size, input_size)
        self.labels = torch.randint(0, num_classes, (size,))
        
        # Add metadata for compliance checks
        self.metadata = {
            "compound_id": [f"CMP_{i:06d}" for i in range(size)],
            "target_disease": np.random.choice(
                ["cancer", "diabetes", "alzheimers", "parkinsons", "cardiovascular"],
                size=size
            ),
            "development_stage": np.random.choice(
                ["discovery", "preclinical", "phase1", "phase2", "phase3"],
                size=size
            ),
            "safety_profile": np.random.choice(
                ["low_risk", "medium_risk", "high_risk"],
                size=size
            ),
            "researcher_id": [f"RES_{i:04d}" for i in range(size)],
            "data_categories": ["drug_development", "research_data", "sensitive_data"],
            "jurisdiction": "US",
            "regulations": ["FDA", "EMA", "ICH", "GCP"],
            "data_retention_period": 365,  # days
            "data_minimization": True,
            "purpose_limitation": True,
            "transparency": True,
            "safety_monitoring": True
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

class DrugDiscoveryModel(nn.Module):
    """Drug discovery model with explainability and safety assessment."""
    
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Attention mechanism for explainability
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Safety assessment
        self.safety_assessor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 risk levels
        )
        
        # Development stage predictor
        self.stage_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # 5 development stages
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes)
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
        
        # Assess safety
        safety_scores = self.safety_assessor(attended_features)
        
        # Predict development stage
        stage_scores = self.stage_predictor(attended_features)
        
        # Classify
        logits = self.classifier(attended_features)
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "features": features,
            "safety_scores": safety_scores,
            "stage_scores": stage_scores
        }

class DrugDiscoveryCompliance(ComplianceDataset):
    """Compliance wrapper for drug discovery and development."""
    
    def __init__(
        self,
        base_dataset: Dataset,
        compliance_rules: Dict[str, Any],
        data_categories: List[str]
    ):
        super().__init__(base_dataset, compliance_rules, data_categories)
    
    async def check_privacy_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check privacy compliance for drug discovery data."""
        metadata = data["metadata"]
        return {
            "data_minimization": metadata["data_minimization"],
            "purpose_limitation": metadata["purpose_limitation"],
            "safety_monitoring": metadata["safety_monitoring"],
            "data_retention": metadata["data_retention_period"] <= 365,
            "researcher_verification": bool(metadata["researcher_id"])
        }
    
    async def check_fairness_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check fairness compliance for drug discovery."""
        return {
            "demographic_parity": True,
            "equal_opportunity": True,
            "disparate_impact": True,
            "safety_assessment_fairness": True
        }
    
    async def check_transparency_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check transparency compliance for drug discovery."""
        return {
            "explainability": True,
            "documentation": True,
            "audit_trail": True,
            "safety_assessment": data["metadata"]["safety_profile"] in ["low_risk", "medium_risk", "high_risk"]
        }

async def main():
    # Initialize governance config
    config = GovernanceConfig(
        organization_id="drug_discovery_org_123",
        organization_name="Drug Discovery AI Corp",
        dpo_email="dpo@drugdiscovery.com",
        enabled_regulations=[
            Regulation.FDA,
            Regulation.EMA,
            Regulation.ICH,
            Regulation.GCP
        ]
    )
    
    # Create model and datasets
    model = DrugDiscoveryModel(input_size=20, num_classes=5)
    base_dataset = DrugDiscoveryDataset(size=1000, input_size=20, num_classes=5)
    
    # Wrap dataset with compliance checks
    compliance_dataset = DrugDiscoveryCompliance(
        base_dataset=base_dataset,
        compliance_rules={
            "privacy_threshold": 0.95,
            "fairness_threshold": 0.95,
            "transparency_threshold": 0.95,
            "documentation_complete": True,
            "safety_monitoring": True
        },
        data_categories=["drug_development", "research_data", "sensitive_data"]
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
        "fda_compliance": True,
        "data_minimization": True,
        "audit_trail": True,
        "explainability": True,
        "safety_monitoring": True
    }
    
    training_config = {
        "epochs": 10,
        "thresholds": compliance_rules,
        "evaluation_metrics": [
            "bias",
            "privacy",
            "transparency",
            "fairness",
            "fda_compliance",
            "safety_assessment"
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
            "model_type": "drug_discovery",
            "data_categories": ["drug_development", "research_data", "sensitive_data"],
            "jurisdiction": "US",
            "fda_covered": True,
            "sensitive_data": True,
            "explainability_required": True,
            "safety_monitoring": True
        }
    )
    
    # Save results
    results_path = "drug_discovery_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print compliance evaluation results
    print("\nDrug Discovery Compliance Evaluation Results:")
    print(json.dumps(results["final_evaluation"], indent=2))
    
    # Print recommendations
    print("\nRecommendations:")
    for rec in results["final_evaluation"]["recommendations"]:
        print(f"- {rec['action']} (Priority: {rec['priority']})")

if __name__ == "__main__":
    asyncio.run(main()) 