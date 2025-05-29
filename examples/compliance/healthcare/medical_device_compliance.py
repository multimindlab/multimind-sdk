"""
Example demonstrating compliance monitoring for medical device AI systems.
This example shows how to ensure FDA compliance and medical ethics
in AI-powered medical device systems.
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

class MedicalDeviceDataset(Dataset):
    """Dataset for medical device data with synthetic data."""
    
    def __init__(self, size: int, input_size: int, num_classes: int):
        self.data = torch.randn(size, input_size)
        self.targets = torch.randint(0, num_classes, (size,))
        self.metadata = self._generate_metadata(size)
    
    def _generate_metadata(self, size: int) -> List[Dict[str, Any]]:
        """Generate synthetic medical device metadata."""
        metadata = []
        
        for _ in range(size):
            calibration_date = datetime.now() - timedelta(days=np.random.randint(0, 30))
            metadata.append({
                "device_id": f"D{np.random.randint(10000, 99999)}",
                "patient_id": f"P{np.random.randint(10000, 99999)}",
                "operator_id": f"O{np.random.randint(100, 999)}",
                "timestamp": datetime.now().isoformat(),
                "device_type": np.random.choice([
                    "monitor", "pump", "ventilator", "defibrillator"
                ]),
                "device_class": np.random.choice([
                    "I", "II", "III"
                ]),
                "measurements": {
                    "value": round(np.random.uniform(0, 100), 2),
                    "unit": np.random.choice(["mmHg", "bpm", "L/min", "J"]),
                    "alarm_threshold": round(np.random.uniform(0, 100), 2)
                },
                "device_status": {
                    "battery_level": np.random.randint(0, 100),
                    "calibration_status": "valid",
                    "last_calibration": calibration_date.isoformat(),
                    "maintenance_due": (calibration_date + timedelta(days=30)).isoformat()
                },
                "data_category": "device_data",
                "consent_status": True,
                "data_retention_period": "10_years",
                "device_metadata": {
                    "manufacturer": np.random.choice([
                        "MedTech", "HealthDev", "LifeCare", "SafeMed"
                    ]),
                    "model": f"MD-{np.random.randint(100, 999)}",
                    "serial_number": f"SN{np.random.randint(10000, 99999)}",
                    "firmware_version": f"v{np.random.randint(1, 5)}",
                    "fda_approved": True,
                    "risk_level": np.random.choice(["low", "medium", "high"]),
                    "quality_control": {
                        "passed": True,
                        "last_check": calibration_date.isoformat(),
                        "next_check": (calibration_date + timedelta(days=30)).isoformat()
                    }
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

class MedicalDeviceModel(nn.Module):
    """Medical device AI model with explainability."""
    
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

class MedicalDeviceCompliance(ComplianceDataset):
    """Healthcare-specific compliance checks for medical devices."""
    
    def _check_privacy(self, item: Dict[str, Any]) -> bool:
        """Check HIPAA privacy compliance."""
        metadata = item["metadata"]
        
        # Check for PHI
        if "patient_id" in metadata:
            if not metadata["patient_id"].startswith("P"):
                return False
        
        # Check device ID
        if not metadata["device_id"].startswith("D"):
            return False
        
        # Check operator ID
        if not metadata["operator_id"].startswith("O"):
            return False
        
        # Check data retention
        if metadata.get("data_retention_period") != "10_years":
            return False
        
        # Check device metadata
        if not all(
            key in metadata["device_metadata"]
            for key in [
                "manufacturer", "model", "serial_number",
                "firmware_version", "fda_approved", "risk_level",
                "quality_control"
            ]
        ):
            return False
        
        return True
    
    def _check_fairness(self, item: Dict[str, Any]) -> bool:
        """Check for healthcare-specific fairness."""
        metadata = item["metadata"]
        
        # Check device type validity
        if metadata["device_type"] not in [
            "monitor", "pump", "ventilator", "defibrillator"
        ]:
            return False
        
        # Check device class validity
        if metadata["device_class"] not in ["I", "II", "III"]:
            return False
        
        # Check measurements validity
        measurements = metadata["measurements"]
        if not (
            0 <= measurements["value"] <= 100 and
            0 <= measurements["alarm_threshold"] <= 100
        ):
            return False
        
        # Check device status validity
        device_status = metadata["device_status"]
        if not (
            0 <= device_status["battery_level"] <= 100 and
            device_status["calibration_status"] == "valid"
        ):
            return False
        
        return True
    
    def _check_transparency(self, item: Dict[str, Any]) -> bool:
        """Check for healthcare-specific transparency."""
        metadata = item["metadata"]
        
        # Check required metadata
        required_fields = [
            "device_id", "patient_id", "operator_id", "timestamp",
            "device_type", "device_class", "measurements",
            "device_status", "device_metadata"
        ]
        if not all(field in metadata for field in required_fields):
            return False
        
        # Check measurements completeness
        if not all(
            key in metadata["measurements"]
            for key in ["value", "unit", "alarm_threshold"]
        ):
            return False
        
        # Check device status completeness
        if not all(
            key in metadata["device_status"]
            for key in [
                "battery_level", "calibration_status",
                "last_calibration", "maintenance_due"
            ]
        ):
            return False
        
        # Check quality control completeness
        if not all(
            key in metadata["device_metadata"]["quality_control"]
            for key in ["passed", "last_check", "next_check"]
        ):
            return False
        
        return True

async def main():
    # Initialize governance config
    config = GovernanceConfig(
        organization_id="health_org_123",
        organization_name="Medical Device AI Corp",
        dpo_email="dpo@meddevice.com",
        enabled_regulations=[
            Regulation.HIPAA,
            Regulation.GDPR,
            Regulation.AI_ACT
        ]
    )
    
    # Create model and datasets
    model = MedicalDeviceModel(input_size=20, num_classes=5)
    base_dataset = MedicalDeviceDataset(size=1000, input_size=20, num_classes=5)
    
    # Wrap dataset with compliance checks
    compliance_dataset = MedicalDeviceCompliance(
        base_dataset=base_dataset,
        compliance_rules={
            "privacy_threshold": 0.9,
            "fairness_threshold": 0.9,
            "transparency_threshold": 0.9,
            "documentation_complete": True
        },
        data_categories=["device_data", "personal_data"]
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
        "model_type": "medical_device",
        "data_categories": ["device_data", "personal_data"],
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
    results_path = "medical_device_results.json"
    trainer.save_training_results(
        results=results,
        path=results_path
    )
    
    # Initialize visualizer
    visualizer = ComplianceVisualizer(results_path)
    
    # Create visualizations
    visualizer.plot_metrics_history(save_path="device_metrics.html")
    visualizer.plot_violations_heatmap(save_path="device_violations.html")
    visualizer.plot_compliance_radar(
        metrics=results["final_evaluation"]["compliance_scores"],
        save_path="device_radar.html"
    )
    visualizer.plot_violation_timeline(save_path="device_timeline.html")
    
    # Create interactive dashboard
    visualizer.create_dashboard(port=8050)
    
    # Print compliance evaluation results
    print("\nMedical Device Compliance Evaluation Results:")
    print(json.dumps(results["final_evaluation"], indent=2))
    
    # Print recommendations
    print("\nRecommendations:")
    for rec in results["final_evaluation"]["recommendations"]:
        print(f"- {rec['action']} (Priority: {rec['priority']})")

if __name__ == "__main__":
    asyncio.run(main()) 