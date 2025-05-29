"""
Example demonstrating compliance monitoring for medical research AI systems.
This example shows how to ensure HIPAA compliance and medical ethics
in AI-powered medical research systems.
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

class MedicalResearchDataset(Dataset):
    """Dataset for medical research data with synthetic data."""
    
    def __init__(self, size: int, input_size: int, num_classes: int):
        self.data = torch.randn(size, input_size)
        self.targets = torch.randint(0, num_classes, (size,))
        self.metadata = self._generate_metadata(size)
    
    def _generate_metadata(self, size: int) -> List[Dict[str, Any]]:
        """Generate synthetic medical research metadata."""
        metadata = []
        study_start = datetime.now() - timedelta(days=365)
        
        for _ in range(size):
            enrollment_date = study_start + timedelta(days=np.random.randint(0, 365))
            metadata.append({
                "participant_id": f"P{np.random.randint(10000, 99999)}",
                "study_id": f"S{np.random.randint(1000, 9999)}",
                "researcher_id": f"R{np.random.randint(100, 999)}",
                "enrollment_date": enrollment_date.isoformat(),
                "study_type": np.random.choice([
                    "observational", "interventional", "registry"
                ]),
                "data_type": np.random.choice([
                    "genomic", "clinical", "imaging", "biomarker"
                ]),
                "consent_status": {
                    "informed_consent": True,
                    "consent_date": enrollment_date.isoformat(),
                    "consent_version": f"v{np.random.randint(1, 5)}",
                    "withdrawal_rights": True
                },
                "data_category": "research_data",
                "data_retention_period": "20_years",
                "research_metadata": {
                    "institution": np.random.choice([
                        "ResearchInst", "MedUniv", "HealthCenter"
                    ]),
                    "irb_approval": True,
                    "irb_number": f"IRB-{np.random.randint(1000, 9999)}",
                    "funding_source": np.random.choice([
                        "NIH", "NSF", "Private", "Foundation"
                    ]),
                    "data_sharing": {
                        "allowed": True,
                        "restrictions": np.random.choice([
                            "none", "deidentified", "restricted"
                        ]),
                        "repository": np.random.choice([
                            "dbGaP", "GEO", "SRA", "Custom"
                        ])
                    },
                    "publication_status": np.random.choice([
                        "planned", "in_progress", "published"
                    ]),
                    "collaborators": [
                        f"Inst{np.random.randint(1, 5)}"
                        for _ in range(np.random.randint(1, 4))
                    ]
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

class MedicalResearchModel(nn.Module):
    """Medical research AI model with explainability."""
    
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
        
        # Ethics monitoring
        self.ethics_monitor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 ethics levels
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        attention_weights = torch.softmax(self.attention(features), dim=1)
        weighted_features = features * attention_weights
        logits = self.classifier(weighted_features)
        ethics_scores = self.ethics_monitor(features)
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "ethics_scores": ethics_scores,
            "features": features
        }

class MedicalResearchCompliance(ComplianceDataset):
    """Healthcare-specific compliance checks for medical research."""
    
    def _check_privacy(self, item: Dict[str, Any]) -> bool:
        """Check HIPAA privacy compliance."""
        metadata = item["metadata"]
        
        # Check for PHI
        if "participant_id" in metadata:
            if not metadata["participant_id"].startswith("P"):
                return False
        
        # Check study ID
        if not metadata["study_id"].startswith("S"):
            return False
        
        # Check researcher ID
        if not metadata["researcher_id"].startswith("R"):
            return False
        
        # Check data retention
        if metadata.get("data_retention_period") != "20_years":
            return False
        
        # Check research metadata
        if not all(
            key in metadata["research_metadata"]
            for key in [
                "institution", "irb_approval", "irb_number",
                "funding_source", "data_sharing", "publication_status",
                "collaborators"
            ]
        ):
            return False
        
        return True
    
    def _check_fairness(self, item: Dict[str, Any]) -> bool:
        """Check for healthcare-specific fairness."""
        metadata = item["metadata"]
        
        # Check study type validity
        if metadata["study_type"] not in [
            "observational", "interventional", "registry"
        ]:
            return False
        
        # Check data type validity
        if metadata["data_type"] not in [
            "genomic", "clinical", "imaging", "biomarker"
        ]:
            return False
        
        # Check consent status
        consent = metadata["consent_status"]
        if not (
            consent["informed_consent"] and
            consent["withdrawal_rights"]
        ):
            return False
        
        # Check data sharing restrictions
        if metadata["research_metadata"]["data_sharing"]["restrictions"] not in [
            "none", "deidentified", "restricted"
        ]:
            return False
        
        return True
    
    def _check_transparency(self, item: Dict[str, Any]) -> bool:
        """Check for healthcare-specific transparency."""
        metadata = item["metadata"]
        
        # Check required metadata
        required_fields = [
            "participant_id", "study_id", "researcher_id",
            "enrollment_date", "study_type", "data_type",
            "consent_status", "research_metadata"
        ]
        if not all(field in metadata for field in required_fields):
            return False
        
        # Check consent status completeness
        if not all(
            key in metadata["consent_status"]
            for key in [
                "informed_consent", "consent_date",
                "consent_version", "withdrawal_rights"
            ]
        ):
            return False
        
        # Check data sharing completeness
        if not all(
            key in metadata["research_metadata"]["data_sharing"]
            for key in ["allowed", "restrictions", "repository"]
        ):
            return False
        
        return True

async def main():
    # Initialize governance config
    config = GovernanceConfig(
        organization_id="health_org_123",
        organization_name="Medical Research AI Corp",
        dpo_email="dpo@medresearch.com",
        enabled_regulations=[
            Regulation.HIPAA,
            Regulation.GDPR,
            Regulation.AI_ACT
        ]
    )
    
    # Create model and datasets
    model = MedicalResearchModel(input_size=20, num_classes=5)
    base_dataset = MedicalResearchDataset(size=1000, input_size=20, num_classes=5)
    
    # Wrap dataset with compliance checks
    compliance_dataset = MedicalResearchCompliance(
        base_dataset=base_dataset,
        compliance_rules={
            "privacy_threshold": 0.9,
            "fairness_threshold": 0.9,
            "transparency_threshold": 0.9,
            "documentation_complete": True
        },
        data_categories=["research_data", "personal_data"]
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
        "model_type": "medical_research",
        "data_categories": ["research_data", "personal_data"],
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
    results_path = "medical_research_results.json"
    trainer.save_training_results(
        results=results,
        path=results_path
    )
    
    # Initialize visualizer
    visualizer = ComplianceVisualizer(results_path)
    
    # Create visualizations
    visualizer.plot_metrics_history(save_path="research_metrics.html")
    visualizer.plot_violations_heatmap(save_path="research_violations.html")
    visualizer.plot_compliance_radar(
        metrics=results["final_evaluation"]["compliance_scores"],
        save_path="research_radar.html"
    )
    visualizer.plot_violation_timeline(save_path="research_timeline.html")
    
    # Create interactive dashboard
    visualizer.create_dashboard(port=8050)
    
    # Print compliance evaluation results
    print("\nMedical Research Compliance Evaluation Results:")
    print(json.dumps(results["final_evaluation"], indent=2))
    
    # Print recommendations
    print("\nRecommendations:")
    for rec in results["final_evaluation"]["recommendations"]:
        print(f"- {rec['action']} (Priority: {rec['priority']})")

if __name__ == "__main__":
    asyncio.run(main()) 