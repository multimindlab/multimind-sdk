"""
Example demonstrating compliance monitoring for clinical trial data analysis.
This example shows how to ensure HIPAA compliance and medical ethics
in AI-powered clinical trial data analysis systems.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Any
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta

from multimind.compliance.model_training import (
    ComplianceDataset,
    ComplianceTrainer,
    ComplianceMetrics
)
from multimind.compliance.visualization import ComplianceVisualizer
from multimind.compliance import GovernanceConfig, Regulation

class ClinicalTrialDataset(Dataset):
    """Dataset for clinical trial data with synthetic data."""
    
    def __init__(self, size: int, input_size: int, num_classes: int):
        self.data = torch.randn(size, input_size)
        self.targets = torch.randint(0, num_classes, (size,))
        self.metadata = self._generate_metadata(size)
    
    def _generate_metadata(self, size: int) -> List[Dict[str, Any]]:
        """Generate synthetic clinical trial metadata."""
        metadata = []
        trial_start = datetime.now() - timedelta(days=365)
        
        for _ in range(size):
            enrollment_date = trial_start + timedelta(days=np.random.randint(0, 365))
            metadata.append({
                "patient_id": f"P{np.random.randint(10000, 99999)}",
                "trial_id": f"T{np.random.randint(1000, 9999)}",
                "site_id": f"S{np.random.randint(100, 999)}",
                "enrollment_date": enrollment_date.isoformat(),
                "visit_number": np.random.randint(1, 10),
                "treatment_arm": np.random.choice([
                    "control", "treatment_a", "treatment_b"
                ]),
                "adverse_events": np.random.choice([True, False]),
                "serious_adverse_events": np.random.choice([True, False]),
                "protocol_deviations": np.random.choice([True, False]),
                "informed_consent": True,
                "data_category": "clinical_trial_data",
                "consent_status": True,
                "data_retention_period": "15_years",
                "trial_metadata": {
                    "phase": np.random.choice(["I", "II", "III", "IV"]),
                    "status": np.random.choice([
                        "active", "completed", "terminated"
                    ]),
                    "sponsor": np.random.choice([
                        "PharmaCorp", "MedResearch", "ClinicalTech"
                    ]),
                    "protocol_version": f"v{np.random.randint(1, 5)}",
                    "irb_approval": True,
                    "data_monitoring_committee": True
                }
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

class ClinicalTrialModel(nn.Module):
    """Clinical trial analysis model with explainability."""
    
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
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
        
        # Safety monitoring
        self.safety_monitor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 safety levels
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        attention_weights = torch.softmax(self.attention(features), dim=1)
        weighted_features = features * attention_weights
        logits = self.classifier(weighted_features)
        safety_scores = self.safety_monitor(features)
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "safety_scores": safety_scores,
            "features": features
        }

class ClinicalTrialCompliance(ComplianceDataset):
    """Healthcare-specific compliance checks for clinical trials."""
    
    def _check_privacy(self, item: Dict[str, Any]) -> bool:
        """Check HIPAA privacy compliance."""
        metadata = item["metadata"]
        
        # Check for PHI
        if "patient_id" in metadata:
            if not metadata["patient_id"].startswith("P"):
                return False
        
        # Check trial ID
        if not metadata["trial_id"].startswith("T"):
            return False
        
        # Check site ID
        if not metadata["site_id"].startswith("S"):
            return False
        
        # Check data retention
        if metadata.get("data_retention_period") != "15_years":
            return False
        
        # Check trial metadata
        if not all(
            key in metadata["trial_metadata"]
            for key in [
                "phase", "status", "sponsor", "protocol_version",
                "irb_approval", "data_monitoring_committee"
            ]
        ):
            return False
        
        return True
    
    def _check_fairness(self, item: Dict[str, Any]) -> bool:
        """Check for healthcare-specific fairness."""
        metadata = item["metadata"]
        
        # Check treatment arm validity
        if metadata["treatment_arm"] not in [
            "control", "treatment_a", "treatment_b"
        ]:
            return False
        
        # Check visit number validity
        if not 1 <= metadata["visit_number"] <= 10:
            return False
        
        # Check trial phase validity
        if metadata["trial_metadata"]["phase"] not in ["I", "II", "III", "IV"]:
            return False
        
        # Check trial status validity
        if metadata["trial_metadata"]["status"] not in [
            "active", "completed", "terminated"
        ]:
            return False
        
        return True
    
    def _check_transparency(self, item: Dict[str, Any]) -> bool:
        """Check for healthcare-specific transparency."""
        metadata = item["metadata"]
        
        # Check required metadata
        required_fields = [
            "patient_id", "trial_id", "site_id", "enrollment_date",
            "visit_number", "treatment_arm", "trial_metadata"
        ]
        if not all(field in metadata for field in required_fields):
            return False
        
        # Check trial metadata completeness
        trial_fields = [
            "phase", "status", "sponsor", "protocol_version",
            "irb_approval", "data_monitoring_committee"
        ]
        if not all(field in metadata["trial_metadata"] for field in trial_fields):
            return False
        
        return True

async def main():
    # Initialize governance config
    config = GovernanceConfig(
        organization_id="health_org_123",
        organization_name="Clinical Trial AI Corp",
        dpo_email="dpo@clinicaltrial.com",
        enabled_regulations=[
            Regulation.HIPAA,
            Regulation.GDPR,
            Regulation.AI_ACT
        ]
    )
    
    # Create model and datasets
    model = ClinicalTrialModel(input_size=20, num_classes=5)
    base_dataset = ClinicalTrialDataset(size=1000, input_size=20, num_classes=5)
    
    # Wrap dataset with compliance checks
    compliance_dataset = ClinicalTrialCompliance(
        base_dataset=base_dataset,
        compliance_rules={
            "privacy_threshold": 0.9,
            "fairness_threshold": 0.9,
            "transparency_threshold": 0.9,
            "documentation_complete": True
        },
        data_categories=["clinical_trial_data", "personal_data"]
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
        "model_type": "clinical_trial",
        "data_categories": ["clinical_trial_data", "personal_data"],
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
    results_path = "clinical_trial_results.json"
    trainer.save_training_results(
        results=results,
        path=results_path
    )
    
    # Initialize visualizer
    visualizer = ComplianceVisualizer(results_path)
    
    # Create visualizations
    visualizer.plot_metrics_history(save_path="trial_metrics.html")
    visualizer.plot_violations_heatmap(save_path="trial_violations.html")
    visualizer.plot_compliance_radar(
        metrics=results["final_evaluation"]["compliance_scores"],
        save_path="trial_radar.html"
    )
    visualizer.plot_violation_timeline(save_path="trial_timeline.html")
    
    # Create interactive dashboard
    visualizer.create_dashboard(port=8050)
    
    # Print compliance evaluation results
    print("\nClinical Trial Compliance Evaluation Results:")
    print(json.dumps(results["final_evaluation"], indent=2))
    
    # Print recommendations
    print("\nRecommendations:")
    for rec in results["final_evaluation"]["recommendations"]:
        print(f"- {rec['action']} (Priority: {rec['priority']})")

if __name__ == "__main__":
    asyncio.run(main()) 