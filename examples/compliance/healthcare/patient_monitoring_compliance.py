"""
Example demonstrating compliance monitoring for patient monitoring systems.
This example shows how to ensure HIPAA compliance and medical ethics
in real-time patient monitoring AI systems.
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

class PatientMonitoringDataset(Dataset):
    """Dataset for patient monitoring with synthetic data."""
    
    def __init__(self, size: int, input_size: int, num_classes: int):
        self.data = torch.randn(size, input_size)
        self.targets = torch.randint(0, num_classes, (size,))
        self.metadata = self._generate_metadata(size)
    
    def _generate_metadata(self, size: int) -> List[Dict[str, Any]]:
        """Generate synthetic patient monitoring metadata."""
        metadata = []
        base_time = datetime.now()
        
        for i in range(size):
            timestamp = base_time + timedelta(minutes=i*5)
            metadata.append({
                "patient_id": f"P{np.random.randint(10000, 99999)}",
                "timestamp": timestamp.isoformat(),
                "device_id": f"D{np.random.randint(1000, 9999)}",
                "vital_signs": {
                    "heart_rate": np.random.randint(60, 100),
                    "blood_pressure": f"{np.random.randint(110, 140)}/{np.random.randint(60, 90)}",
                    "oxygen_saturation": np.random.randint(95, 100),
                    "temperature": round(np.random.uniform(36.5, 37.5), 1)
                },
                "location": np.random.choice(["ICU", "ER", "General", "Home"]),
                "monitoring_type": np.random.choice(["continuous", "intermittent"]),
                "data_category": "health_data",
                "consent_status": True,
                "data_retention_period": "7_years",
                "alert_thresholds": {
                    "heart_rate": {"min": 60, "max": 100},
                    "blood_pressure": {"min": 90, "max": 140},
                    "oxygen_saturation": {"min": 95, "max": 100},
                    "temperature": {"min": 36.5, "max": 37.5}
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

class PatientMonitoringModel(nn.Module):
    """Patient monitoring model with real-time alerting."""
    
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
        
        # Alert mechanism
        self.alert_threshold = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        alert_prob = self.alert_threshold(features)
        
        return {
            "logits": logits,
            "alert_probability": alert_prob,
            "features": features
        }

class PatientMonitoringCompliance(ComplianceDataset):
    """Healthcare-specific compliance checks for patient monitoring."""
    
    def _check_privacy(self, item: Dict[str, Any]) -> bool:
        """Check HIPAA privacy compliance."""
        metadata = item["metadata"]
        
        # Check for PHI
        if "patient_id" in metadata:
            if not metadata["patient_id"].startswith("P"):
                return False
        
        # Check device security
        if not metadata["device_id"].startswith("D"):
            return False
        
        # Check data retention
        if metadata.get("data_retention_period") != "7_years":
            return False
        
        # Check monitoring type compliance
        if metadata["monitoring_type"] == "continuous":
            if not self.compliance_rules.get("continuous_monitoring_approved", False):
                return False
        
        return True
    
    def _check_fairness(self, item: Dict[str, Any]) -> bool:
        """Check for healthcare-specific fairness."""
        metadata = item["metadata"]
        
        # Check location-based fairness
        if metadata["location"] not in ["ICU", "ER", "General", "Home"]:
            return False
        
        # Check monitoring type fairness
        if metadata["monitoring_type"] not in ["continuous", "intermittent"]:
            return False
        
        # Check vital signs thresholds
        vital_signs = metadata["vital_signs"]
        thresholds = metadata["alert_thresholds"]
        
        for vital, value in vital_signs.items():
            if isinstance(value, str):  # Handle blood pressure
                systolic, diastolic = map(int, value.split("/"))
                if (systolic < thresholds[vital]["min"] or 
                    systolic > thresholds[vital]["max"] or
                    diastolic < thresholds[vital]["min"] or
                    diastolic > thresholds[vital]["max"]):
                    return False
            else:
                if (value < thresholds[vital]["min"] or 
                    value > thresholds[vital]["max"]):
                    return False
        
        return True
    
    def _check_transparency(self, item: Dict[str, Any]) -> bool:
        """Check for healthcare-specific transparency."""
        metadata = item["metadata"]
        
        # Check required metadata
        required_fields = [
            "patient_id", "timestamp", "device_id", "vital_signs",
            "location", "monitoring_type", "alert_thresholds"
        ]
        if not all(field in metadata for field in required_fields):
            return False
        
        # Check alert thresholds documentation
        if not all(
            vital in metadata["alert_thresholds"]
            for vital in metadata["vital_signs"].keys()
        ):
            return False
        
        return True

async def main():
    # Initialize governance config
    config = GovernanceConfig(
        organization_id="health_org_123",
        organization_name="Patient Monitoring AI Corp",
        dpo_email="dpo@patientmonitoring.com",
        enabled_regulations=[
            Regulation.HIPAA,
            Regulation.GDPR,
            Regulation.AI_ACT
        ]
    )
    
    # Create model and datasets
    model = PatientMonitoringModel(input_size=20, num_classes=5)
    base_dataset = PatientMonitoringDataset(size=1000, input_size=20, num_classes=5)
    
    # Wrap dataset with compliance checks
    compliance_dataset = PatientMonitoringCompliance(
        base_dataset=base_dataset,
        compliance_rules={
            "privacy_threshold": 0.9,
            "fairness_threshold": 0.9,
            "transparency_threshold": 0.9,
            "continuous_monitoring_approved": True,
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
        "real_time_monitoring": True
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
        "model_type": "patient_monitoring",
        "data_categories": ["health_data", "personal_data"],
        "jurisdiction": "US",
        "hipaa_covered": True,
        "sensitive_data": True,
        "real_time_required": True
    }
    
    results = await trainer.train(
        train_data=train_loader,
        val_data=val_loader,
        metadata=metadata
    )
    
    # Save training results
    results_path = "patient_monitoring_results.json"
    trainer.save_training_results(
        results=results,
        path=results_path
    )
    
    # Initialize visualizer
    visualizer = ComplianceVisualizer(results_path)
    
    # Create visualizations
    visualizer.plot_metrics_history(save_path="monitoring_metrics.html")
    visualizer.plot_violations_heatmap(save_path="monitoring_violations.html")
    visualizer.plot_compliance_radar(
        metrics=results["final_evaluation"]["compliance_scores"],
        save_path="monitoring_radar.html"
    )
    visualizer.plot_violation_timeline(save_path="monitoring_timeline.html")
    
    # Create interactive dashboard
    visualizer.create_dashboard(port=8050)
    
    # Print compliance evaluation results
    print("\nPatient Monitoring Compliance Evaluation Results:")
    print(json.dumps(results["final_evaluation"], indent=2))
    
    # Print recommendations
    print("\nRecommendations:")
    for rec in results["final_evaluation"]["recommendations"]:
        print(f"- {rec['action']} (Priority: {rec['priority']})")

if __name__ == "__main__":
    asyncio.run(main()) 