"""
Example demonstrating compliance monitoring for medical imaging models.
This example shows how to ensure HIPAA compliance and medical ethics
in AI-powered medical imaging systems.
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

class MedicalImagingDataset(Dataset):
    """Dataset for medical imaging with synthetic data."""
    
    def __init__(self, size: int, input_size: int, num_classes: int):
        self.data = torch.randn(size, input_size)
        self.targets = torch.randint(0, num_classes, (size,))
        self.metadata = self._generate_metadata(size)
    
    def _generate_metadata(self, size: int) -> List[Dict[str, Any]]:
        """Generate synthetic medical imaging metadata."""
        metadata = []
        for _ in range(size):
            metadata.append({
                "patient_id": f"P{np.random.randint(10000, 99999)}",
                "study_id": f"S{np.random.randint(1000, 9999)}",
                "image_type": np.random.choice([
                    "X-ray", "CT", "MRI", "Ultrasound"
                ]),
                "body_part": np.random.choice([
                    "chest", "brain", "abdomen", "spine"
                ]),
                "image_quality": np.random.choice([
                    "high", "medium", "low"
                ]),
                "contrast_used": np.random.choice([True, False]),
                "radiation_dose": round(np.random.uniform(0.1, 5.0), 2),
                "acquisition_date": datetime.now().isoformat(),
                "radiologist_id": f"R{np.random.randint(100, 999)}",
                "data_category": "health_data",
                "consent_status": True,
                "data_retention_period": "7_years",
                "dicom_metadata": {
                    "modality": np.random.choice(["DX", "CT", "MR", "US"]),
                    "manufacturer": np.random.choice([
                        "GE", "Siemens", "Philips", "Toshiba"
                    ]),
                    "pixel_spacing": [0.5, 0.5],
                    "slice_thickness": round(np.random.uniform(0.5, 5.0), 1)
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

class MedicalImagingModel(nn.Module):
    """Medical imaging model with explainability."""
    
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
        
        # Quality assessment
        self.quality_assessor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 quality levels
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        attention_weights = torch.softmax(self.attention(features), dim=1)
        weighted_features = features * attention_weights
        logits = self.classifier(weighted_features)
        quality_scores = self.quality_assessor(features)
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "quality_scores": quality_scores,
            "features": features
        }

class MedicalImagingCompliance(ComplianceDataset):
    """Healthcare-specific compliance checks for medical imaging."""
    
    def _check_privacy(self, item: Dict[str, Any]) -> bool:
        """Check HIPAA privacy compliance."""
        metadata = item["metadata"]
        
        # Check for PHI
        if "patient_id" in metadata:
            if not metadata["patient_id"].startswith("P"):
                return False
        
        # Check study ID
        if not metadata["study_id"].startswith("S"):
            return False
        
        # Check radiologist ID
        if not metadata["radiologist_id"].startswith("R"):
            return False
        
        # Check data retention
        if metadata.get("data_retention_period") != "7_years":
            return False
        
        # Check DICOM metadata
        if not all(
            key in metadata["dicom_metadata"]
            for key in ["modality", "manufacturer", "pixel_spacing", "slice_thickness"]
        ):
            return False
        
        return True
    
    def _check_fairness(self, item: Dict[str, Any]) -> bool:
        """Check for healthcare-specific fairness."""
        metadata = item["metadata"]
        
        # Check image type validity
        if metadata["image_type"] not in ["X-ray", "CT", "MRI", "Ultrasound"]:
            return False
        
        # Check body part validity
        if metadata["body_part"] not in ["chest", "brain", "abdomen", "spine"]:
            return False
        
        # Check image quality
        if metadata["image_quality"] not in ["high", "medium", "low"]:
            return False
        
        # Check radiation dose for X-ray and CT
        if metadata["image_type"] in ["X-ray", "CT"]:
            if metadata["radiation_dose"] <= 0 or metadata["radiation_dose"] > 5.0:
                return False
        
        return True
    
    def _check_transparency(self, item: Dict[str, Any]) -> bool:
        """Check for healthcare-specific transparency."""
        metadata = item["metadata"]
        
        # Check required metadata
        required_fields = [
            "patient_id", "study_id", "image_type", "body_part",
            "image_quality", "acquisition_date", "radiologist_id",
            "dicom_metadata"
        ]
        if not all(field in metadata for field in required_fields):
            return False
        
        # Check DICOM metadata completeness
        dicom_fields = ["modality", "manufacturer", "pixel_spacing", "slice_thickness"]
        if not all(field in metadata["dicom_metadata"] for field in dicom_fields):
            return False
        
        return True

async def main():
    # Initialize governance config
    config = GovernanceConfig(
        organization_id="health_org_123",
        organization_name="Medical Imaging AI Corp",
        dpo_email="dpo@medicalimaging.com",
        enabled_regulations=[
            Regulation.HIPAA,
            Regulation.GDPR,
            Regulation.AI_ACT
        ]
    )
    
    # Create model and datasets
    model = MedicalImagingModel(input_size=20, num_classes=5)
    base_dataset = MedicalImagingDataset(size=1000, input_size=20, num_classes=5)
    
    # Wrap dataset with compliance checks
    compliance_dataset = MedicalImagingCompliance(
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
        "model_type": "medical_imaging",
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
    results_path = "medical_imaging_results.json"
    trainer.save_training_results(
        results=results,
        path=results_path
    )
    
    # Initialize visualizer
    visualizer = ComplianceVisualizer(results_path)
    
    # Create visualizations
    visualizer.plot_metrics_history(save_path="imaging_metrics.html")
    visualizer.plot_violations_heatmap(save_path="imaging_violations.html")
    visualizer.plot_compliance_radar(
        metrics=results["final_evaluation"]["compliance_scores"],
        save_path="imaging_radar.html"
    )
    visualizer.plot_violation_timeline(save_path="imaging_timeline.html")
    
    # Create interactive dashboard
    visualizer.create_dashboard(port=8050)
    
    # Print compliance evaluation results
    print("\nMedical Imaging Compliance Evaluation Results:")
    print(json.dumps(results["final_evaluation"], indent=2))
    
    # Print recommendations
    print("\nRecommendations:")
    for rec in results["final_evaluation"]["recommendations"]:
        print(f"- {rec['action']} (Priority: {rec['priority']})")

if __name__ == "__main__":
    asyncio.run(main()) 