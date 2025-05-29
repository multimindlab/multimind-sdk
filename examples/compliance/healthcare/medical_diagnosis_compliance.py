"""
Example demonstrating compliance monitoring for medical diagnosis models.
This example shows how to ensure HIPAA compliance and medical ethics
in diagnostic AI systems.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Any
import asyncio
import json
from pathlib import Path
from datetime import datetime

from multimind.compliance.model_training import (
    ComplianceDataset,
    ComplianceTrainer,
    ComplianceMetrics
)
from multimind.compliance.visualization import ComplianceVisualizer
from multimind.compliance import GovernanceConfig, Regulation

class MedicalDiagnosisDataset(Dataset):
    """Dataset for medical diagnosis with synthetic data."""
    
    def __init__(self, size: int, input_size: int, num_classes: int):
        self.data = torch.randn(size, input_size)
        self.targets = torch.randint(0, num_classes, (size,))
        self.metadata = self._generate_metadata(size)
    
    def _generate_metadata(self, size: int) -> List[Dict[str, Any]]:
        """Generate synthetic patient metadata."""
        metadata = []
        for _ in range(size):
            metadata.append({
                "patient_id": f"P{np.random.randint(10000, 99999)}",
                "age": np.random.randint(18, 90),
                "gender": np.random.choice(["M", "F"]),
                "medical_history": np.random.choice([
                    "hypertension", "diabetes", "heart_disease", "none"
                ]),
                "sensitive_conditions": np.random.choice([True, False], p=[0.1, 0.9]),
                "data_category": "health_data",
                "consent_status": np.random.choice([True, False], p=[0.9, 0.1]),
                "data_retention_period": "7_years"
            })
        return metadata
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "input": self.data[idx],
            "target": self.targets[idx],
            "metadata": self.metadata[idx]
        }
    
    def __len__(self) -> int:
        return len(self.data)

class DiagnosisModel(nn.Module):
    """Medical diagnosis model with explainability."""
    
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Linear(64, num_classes)
        
        # Attention mechanism for explainability
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        attention_weights = torch.softmax(self.attention(features), dim=1)
        weighted_features = features * attention_weights
        logits = self.classifier(weighted_features)
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "features": features
        }

class MedicalDiagnosisCompliance(ComplianceDataset):
    """Healthcare-specific compliance checks for diagnosis models."""
    
    def _check_privacy(self, item: Dict[str, Any]) -> bool:
        """Check HIPAA privacy compliance."""
        metadata = item["metadata"]
        
        # Check for PHI
        if "patient_id" in metadata:
            if not metadata["patient_id"].startswith("P"):
                return False
        
        # Check consent
        if not metadata.get("consent_status", False):
            return False
        
        # Check data retention
        if metadata.get("data_retention_period") != "7_years":
            return False
        
        # Check sensitive data handling
        if metadata.get("sensitive_conditions", False):
            if not self.compliance_rules.get("handle_sensitive_data", False):
                return False
        
        return True
    
    def _check_fairness(self, item: Dict[str, Any]) -> bool:
        """Check for healthcare-specific fairness."""
        metadata = item["metadata"]
        
        # Check age-based fairness
        if metadata["age"] < 18 or metadata["age"] > 90:
            return False
        
        # Check gender-based fairness
        if metadata["gender"] not in ["M", "F"]:
            return False
        
        # Check medical history bias
        if metadata["medical_history"] == "none" and np.random.random() < 0.1:
            return False
        
        return True
    
    def _check_transparency(self, item: Dict[str, Any]) -> bool:
        """Check for healthcare-specific transparency."""
        metadata = item["metadata"]
        
        # Check required metadata
        required_fields = [
            "patient_id", "age", "gender", "medical_history",
            "consent_status", "data_retention_period"
        ]
        if not all(field in metadata for field in required_fields):
            return False
        
        # Check documentation
        if not self.compliance_rules.get("documentation_complete", False):
            return False
        
        return True

async def main():
    # Initialize governance config
    config = GovernanceConfig(
        organization_id="health_org_123",
        organization_name="Medical AI Corp",
        dpo_email="dpo@medicalai.com",
        enabled_regulations=[
            Regulation.HIPAA,
            Regulation.GDPR,
            Regulation.AI_ACT
        ]
    )
    
    # Create model and datasets
    model = DiagnosisModel(input_size=20, num_classes=5)
    base_dataset = MedicalDiagnosisDataset(size=1000, input_size=20, num_classes=5)
    
    # Wrap dataset with compliance checks
    compliance_dataset = MedicalDiagnosisCompliance(
        base_dataset=base_dataset,
        compliance_rules={
            "privacy_threshold": 0.9,
            "fairness_threshold": 0.9,
            "transparency_threshold": 0.9,
            "handle_sensitive_data": True,
            "documentation_complete": True
        },
        data_categories=["health_data", "personal_data"]
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
    metadata = {
        "model_type": "medical_diagnosis",
        "data_categories": ["health_data", "personal_data"],
        "jurisdiction": "US",
        "hipaa_covered": True,
        "sensitive_data": True,
        "explainability_required": True
    }
    
    results = await trainer.train(
        train_data=train_loader,
        val_data=val_loader,
        metadata=metadata
    )
    
    # Save training results
    results_path = "medical_diagnosis_results.json"
    trainer.save_training_results(
        results=results,
        path=results_path
    )
    
    # Initialize visualizer
    visualizer = ComplianceVisualizer(results_path)
    
    # Create visualizations
    visualizer.plot_metrics_history(save_path="diagnosis_metrics.html")
    visualizer.plot_violations_heatmap(save_path="diagnosis_violations.html")
    visualizer.plot_compliance_radar(
        metrics=results["final_evaluation"]["compliance_scores"],
        save_path="diagnosis_radar.html"
    )
    visualizer.plot_violation_timeline(save_path="diagnosis_timeline.html")
    
    # Create interactive dashboard
    visualizer.create_dashboard(port=8050)
    
    # Print compliance evaluation results
    print("\nMedical Diagnosis Compliance Evaluation Results:")
    print(json.dumps(results["final_evaluation"], indent=2))
    
    # Print recommendations
    print("\nRecommendations:")
    for rec in results["final_evaluation"]["recommendations"]:
        print(f"- {rec['action']} (Priority: {rec['priority']})")

if __name__ == "__main__":
    asyncio.run(main()) 