"""
Example demonstrating compliance monitoring for electronic health records (EHR).
This example shows how to ensure HIPAA compliance and medical ethics
in AI-powered EHR analysis systems.
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

class EHRDataset(Dataset):
    """Dataset for electronic health records with synthetic data."""
    
    def __init__(self, size: int, input_size: int, num_classes: int):
        self.data = torch.randn(size, input_size)
        self.targets = torch.randint(0, num_classes, (size,))
        self.metadata = self._generate_metadata(size)
    
    def _generate_metadata(self, size: int) -> List[Dict[str, Any]]:
        """Generate synthetic EHR metadata."""
        metadata = []
        
        for _ in range(size):
            visit_date = datetime.now() - timedelta(days=np.random.randint(0, 365))
            metadata.append({
                "patient_id": f"P{np.random.randint(10000, 99999)}",
                "visit_id": f"V{np.random.randint(1000, 9999)}",
                "provider_id": f"PR{np.random.randint(100, 999)}",
                "visit_date": visit_date.isoformat(),
                "visit_type": np.random.choice([
                    "routine", "emergency", "follow_up", "specialist"
                ]),
                "diagnosis_codes": [
                    f"ICD-10-{np.random.randint(10000, 99999)}"
                    for _ in range(np.random.randint(1, 4))
                ],
                "medications": [
                    f"MED-{np.random.randint(1000, 9999)}"
                    for _ in range(np.random.randint(0, 5))
                ],
                "lab_results": {
                    "test_id": f"LAB-{np.random.randint(100, 999)}",
                    "result": round(np.random.uniform(0, 100), 2),
                    "unit": np.random.choice(["mg/dL", "mmol/L", "g/L"])
                },
                "vital_signs": {
                    "blood_pressure": f"{np.random.randint(90, 140)}/{np.random.randint(60, 90)}",
                    "heart_rate": np.random.randint(60, 100),
                    "temperature": round(np.random.uniform(36.1, 37.2), 1),
                    "oxygen_saturation": np.random.randint(95, 100)
                },
                "data_category": "health_data",
                "consent_status": True,
                "data_retention_period": "7_years",
                "ehr_metadata": {
                    "system": np.random.choice([
                        "Epic", "Cerner", "Allscripts", "Meditech"
                    ]),
                    "version": f"v{np.random.randint(1, 5)}",
                    "encryption_level": "AES-256",
                    "access_level": np.random.choice([
                        "full", "limited", "read_only"
                    ]),
                    "audit_trail": True,
                    "backup_status": "encrypted"
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

class EHRModel(nn.Module):
    """EHR analysis model with explainability."""
    
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
        
        # Risk assessment
        self.risk_assessor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 risk levels
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        attention_weights = torch.softmax(self.attention(features), dim=1)
        weighted_features = features * attention_weights
        logits = self.classifier(weighted_features)
        risk_scores = self.risk_assessor(features)
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "risk_scores": risk_scores,
            "features": features
        }

class EHRCompliance(ComplianceDataset):
    """Healthcare-specific compliance checks for EHR."""
    
    def _check_privacy(self, item: Dict[str, Any]) -> bool:
        """Check HIPAA privacy compliance."""
        metadata = item["metadata"]
        
        # Check for PHI
        if "patient_id" in metadata:
            if not metadata["patient_id"].startswith("P"):
                return False
        
        # Check visit ID
        if not metadata["visit_id"].startswith("V"):
            return False
        
        # Check provider ID
        if not metadata["provider_id"].startswith("PR"):
            return False
        
        # Check data retention
        if metadata.get("data_retention_period") != "7_years":
            return False
        
        # Check EHR metadata
        if not all(
            key in metadata["ehr_metadata"]
            for key in [
                "system", "version", "encryption_level",
                "access_level", "audit_trail", "backup_status"
            ]
        ):
            return False
        
        return True
    
    def _check_fairness(self, item: Dict[str, Any]) -> bool:
        """Check for healthcare-specific fairness."""
        metadata = item["metadata"]
        
        # Check visit type validity
        if metadata["visit_type"] not in [
            "routine", "emergency", "follow_up", "specialist"
        ]:
            return False
        
        # Check diagnosis codes format
        if not all(
            code.startswith("ICD-10-") and code[7:].isdigit()
            for code in metadata["diagnosis_codes"]
        ):
            return False
        
        # Check medication format
        if not all(
            med.startswith("MED-") and med[4:].isdigit()
            for med in metadata["medications"]
        ):
            return False
        
        # Check vital signs ranges
        vital_signs = metadata["vital_signs"]
        bp = vital_signs["blood_pressure"].split("/")
        if not (
            90 <= int(bp[0]) <= 140 and
            60 <= int(bp[1]) <= 90 and
            60 <= vital_signs["heart_rate"] <= 100 and
            36.1 <= vital_signs["temperature"] <= 37.2 and
            95 <= vital_signs["oxygen_saturation"] <= 100
        ):
            return False
        
        return True
    
    def _check_transparency(self, item: Dict[str, Any]) -> bool:
        """Check for healthcare-specific transparency."""
        metadata = item["metadata"]
        
        # Check required metadata
        required_fields = [
            "patient_id", "visit_id", "provider_id", "visit_date",
            "visit_type", "diagnosis_codes", "medications",
            "lab_results", "vital_signs", "ehr_metadata"
        ]
        if not all(field in metadata for field in required_fields):
            return False
        
        # Check lab results completeness
        if not all(
            key in metadata["lab_results"]
            for key in ["test_id", "result", "unit"]
        ):
            return False
        
        # Check vital signs completeness
        if not all(
            key in metadata["vital_signs"]
            for key in [
                "blood_pressure", "heart_rate",
                "temperature", "oxygen_saturation"
            ]
        ):
            return False
        
        return True

async def main():
    # Initialize governance config
    config = GovernanceConfig(
        organization_id="health_org_123",
        organization_name="EHR AI Corp",
        dpo_email="dpo@ehrai.com",
        enabled_regulations=[
            Regulation.HIPAA,
            Regulation.GDPR,
            Regulation.AI_ACT
        ]
    )
    
    # Create model and datasets
    model = EHRModel(input_size=20, num_classes=5)
    base_dataset = EHRDataset(size=1000, input_size=20, num_classes=5)
    
    # Wrap dataset with compliance checks
    compliance_dataset = EHRCompliance(
        base_dataset=base_dataset,
        compliance_rules={
            "privacy_threshold": 0.9,
            "fairness_threshold": 0.9,
            "transparency_threshold": 0.9,
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
        "model_type": "ehr",
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
    results_path = "ehr_results.json"
    trainer.save_training_results(
        results=results,
        path=results_path
    )
    
    # Initialize visualizer
    visualizer = ComplianceVisualizer(results_path)
    
    # Create visualizations
    visualizer.plot_metrics_history(save_path="ehr_metrics.html")
    visualizer.plot_violations_heatmap(save_path="ehr_violations.html")
    visualizer.plot_compliance_radar(
        metrics=results["final_evaluation"]["compliance_scores"],
        save_path="ehr_radar.html"
    )
    visualizer.plot_violation_timeline(save_path="ehr_timeline.html")
    
    # Create interactive dashboard
    visualizer.create_dashboard(port=8050)
    
    # Print compliance evaluation results
    print("\nEHR Compliance Evaluation Results:")
    print(json.dumps(results["final_evaluation"], indent=2))
    
    # Print recommendations
    print("\nRecommendations:")
    for rec in results["final_evaluation"]["recommendations"]:
        print(f"- {rec['action']} (Priority: {rec['priority']})")

if __name__ == "__main__":
    asyncio.run(main()) 