"""
Example demonstrating compliance monitoring for telemedicine AI systems.
This example shows how to ensure HIPAA compliance and medical ethics
in AI-powered telemedicine systems.
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

class TelemedicineDataset(Dataset):
    """Dataset for telemedicine data with synthetic data."""
    
    def __init__(self, size: int, input_size: int, num_classes: int):
        self.data = torch.randn(size, input_size)
        self.targets = torch.randint(0, num_classes, (size,))
        self.metadata = self._generate_metadata(size)
    
    def _generate_metadata(self, size: int) -> List[Dict[str, Any]]:
        """Generate synthetic telemedicine metadata."""
        metadata = []
        
        for _ in range(size):
            session_start = datetime.now() - timedelta(minutes=np.random.randint(0, 60))
            metadata.append({
                "patient_id": f"P{np.random.randint(10000, 99999)}",
                "provider_id": f"PR{np.random.randint(100, 999)}",
                "session_id": f"S{np.random.randint(1000, 9999)}",
                "session_start": session_start.isoformat(),
                "session_end": (session_start + timedelta(minutes=30)).isoformat(),
                "session_type": np.random.choice([
                    "video", "audio", "chat", "hybrid"
                ]),
                "platform": np.random.choice([
                    "Zoom", "Teams", "Doxy", "Custom"
                ]),
                "connection_quality": np.random.choice([
                    "excellent", "good", "fair", "poor"
                ]),
                "data_category": "telehealth_data",
                "consent_status": True,
                "data_retention_period": "7_years",
                "session_metadata": {
                    "encryption": "AES-256",
                    "authentication": "2FA",
                    "recording_status": np.random.choice([True, False]),
                    "recording_consent": True,
                    "transcription_status": np.random.choice([True, False]),
                    "transcription_consent": True,
                    "data_transfer": {
                        "encrypted": True,
                        "protocol": "HTTPS",
                        "storage": "HIPAA_compliant"
                    },
                    "emergency_protocol": {
                        "activated": False,
                        "location_shared": False,
                        "emergency_contacts": [
                            f"EC{np.random.randint(1, 5)}"
                            for _ in range(np.random.randint(1, 3))
                        ]
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

class TelemedicineModel(nn.Module):
    """Telemedicine AI model with explainability."""
    
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
        
        # Quality monitoring
        self.quality_monitor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 quality levels
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        attention_weights = torch.softmax(self.attention(features), dim=1)
        weighted_features = features * attention_weights
        logits = self.classifier(weighted_features)
        quality_scores = self.quality_monitor(features)
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "quality_scores": quality_scores,
            "features": features
        }

class TelemedicineCompliance(ComplianceDataset):
    """Healthcare-specific compliance checks for telemedicine."""
    
    def _check_privacy(self, item: Dict[str, Any]) -> bool:
        """Check HIPAA privacy compliance."""
        metadata = item["metadata"]
        
        # Check for PHI
        if "patient_id" in metadata:
            if not metadata["patient_id"].startswith("P"):
                return False
        
        # Check provider ID
        if not metadata["provider_id"].startswith("PR"):
            return False
        
        # Check session ID
        if not metadata["session_id"].startswith("S"):
            return False
        
        # Check data retention
        if metadata.get("data_retention_period") != "7_years":
            return False
        
        # Check session metadata
        if not all(
            key in metadata["session_metadata"]
            for key in [
                "encryption", "authentication", "recording_status",
                "recording_consent", "transcription_status",
                "transcription_consent", "data_transfer",
                "emergency_protocol"
            ]
        ):
            return False
        
        return True
    
    def _check_fairness(self, item: Dict[str, Any]) -> bool:
        """Check for healthcare-specific fairness."""
        metadata = item["metadata"]
        
        # Check session type validity
        if metadata["session_type"] not in [
            "video", "audio", "chat", "hybrid"
        ]:
            return False
        
        # Check platform validity
        if metadata["platform"] not in [
            "Zoom", "Teams", "Doxy", "Custom"
        ]:
            return False
        
        # Check connection quality validity
        if metadata["connection_quality"] not in [
            "excellent", "good", "fair", "poor"
        ]:
            return False
        
        # Check recording consent
        if metadata["session_metadata"]["recording_status"]:
            if not metadata["session_metadata"]["recording_consent"]:
                return False
        
        # Check transcription consent
        if metadata["session_metadata"]["transcription_status"]:
            if not metadata["session_metadata"]["transcription_consent"]:
                return False
        
        return True
    
    def _check_transparency(self, item: Dict[str, Any]) -> bool:
        """Check for healthcare-specific transparency."""
        metadata = item["metadata"]
        
        # Check required metadata
        required_fields = [
            "patient_id", "provider_id", "session_id",
            "session_start", "session_end", "session_type",
            "platform", "connection_quality", "session_metadata"
        ]
        if not all(field in metadata for field in required_fields):
            return False
        
        # Check data transfer completeness
        if not all(
            key in metadata["session_metadata"]["data_transfer"]
            for key in ["encrypted", "protocol", "storage"]
        ):
            return False
        
        # Check emergency protocol completeness
        if not all(
            key in metadata["session_metadata"]["emergency_protocol"]
            for key in [
                "activated", "location_shared",
                "emergency_contacts"
            ]
        ):
            return False
        
        return True

async def main():
    # Initialize governance config
    config = GovernanceConfig(
        organization_id="health_org_123",
        organization_name="Telemedicine AI Corp",
        dpo_email="dpo@telemedicine.com",
        enabled_regulations=[
            Regulation.HIPAA,
            Regulation.GDPR,
            Regulation.AI_ACT
        ]
    )
    
    # Create model and datasets
    model = TelemedicineModel(input_size=20, num_classes=5)
    base_dataset = TelemedicineDataset(size=1000, input_size=20, num_classes=5)
    
    # Wrap dataset with compliance checks
    compliance_dataset = TelemedicineCompliance(
        base_dataset=base_dataset,
        compliance_rules={
            "privacy_threshold": 0.9,
            "fairness_threshold": 0.9,
            "transparency_threshold": 0.9,
            "documentation_complete": True
        },
        data_categories=["telehealth_data", "personal_data"]
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
        "model_type": "telemedicine",
        "data_categories": ["telehealth_data", "personal_data"],
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
    results_path = "telemedicine_results.json"
    trainer.save_training_results(
        results=results,
        path=results_path
    )
    
    # Initialize visualizer
    visualizer = ComplianceVisualizer(results_path)
    
    # Create visualizations
    visualizer.plot_metrics_history(save_path="telemedicine_metrics.html")
    visualizer.plot_violations_heatmap(save_path="telemedicine_violations.html")
    visualizer.plot_compliance_radar(
        metrics=results["final_evaluation"]["compliance_scores"],
        save_path="telemedicine_radar.html"
    )
    visualizer.plot_violation_timeline(save_path="telemedicine_timeline.html")
    
    # Create interactive dashboard
    visualizer.create_dashboard(port=8050)
    
    # Print compliance evaluation results
    print("\nTelemedicine Compliance Evaluation Results:")
    print(json.dumps(results["final_evaluation"], indent=2))
    
    # Print recommendations
    print("\nRecommendations:")
    for rec in results["final_evaluation"]["recommendations"]:
        print(f"- {rec['action']} (Priority: {rec['priority']})")

if __name__ == "__main__":
    asyncio.run(main()) 