"""
Example demonstrating model finetuning with compliance monitoring.
This example shows how to:
1. Finetune a model while monitoring compliance
2. Apply differential privacy during training
3. Generate compliance proofs
4. Track model changes and watermarks
5. Ensure regulatory compliance during training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime

from multimind.compliance.advanced import (
    ComplianceShard,
    SelfHealingCompliance,
    ExplainableDTO,
    ModelWatermarking,
    AdaptivePrivacy,
    RegulatoryChangeDetector,
    FederatedCompliance,
    ComplianceLevel
)
from multimind.compliance.advanced_config import (
    ComplianceShardConfig,
    SelfHealingConfig,
    ExplainableDTOConfig,
    ModelWatermarkingConfig,
    AdaptivePrivacyConfig,
    RegulatoryChangeConfig,
    FederatedComplianceConfig
)

class CompliantModelTrainer:
    """Trainer that ensures compliance during model finetuning."""
    
    def __init__(
        self,
        model_name: str,
        compliance_config: Dict[str, Any],
        privacy_config: Optional[Dict[str, Any]] = None,
        watermark_config: Optional[Dict[str, Any]] = None
    ):
        # Initialize model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize compliance components
        self.compliance_shard = ComplianceShard(
            shard_id="training_shard",
            jurisdiction="global",
            config=compliance_config
        )
        
        # Initialize privacy mechanism
        self.privacy = AdaptivePrivacy(
            privacy_config or {
                "initial_epsilon": 1.0,
                "min_epsilon": 0.1,
                "max_epsilon": 10.0,
                "adaptation_rate": 0.1,
                "feedback_window": 100
            }
        )
        
        # Initialize watermarking
        self.watermarking = ModelWatermarking(
            watermark_config or {
                "watermark_type": "invisible",
                "fingerprint_size": 256,
                "tracking_enabled": True,
                "verification_threshold": 0.9
            }
        )
        
        # Initialize explainable DTO
        self.explainer = ExplainableDTO({
            "model_version": "1.0.0",
            "confidence_threshold": 0.8,
            "explanation_depth": 3,
            "include_metadata": True
        })
        
        # Training history
        self.training_history = []
        self.compliance_history = []
    
    async def finetune(
        self,
        train_data: List[Dict[str, Any]],
        val_data: Optional[List[Dict[str, Any]]] = None,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ):
        """Finetune model while ensuring compliance."""
        # Prepare data
        train_dataloader = self._prepare_dataloader(train_data, batch_size)
        val_dataloader = self._prepare_dataloader(val_data, batch_size) if val_data else None
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch in train_dataloader:
                # Forward pass with privacy
                outputs = await self._private_forward(batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                
                # Check compliance
                compliance_result = await self._check_compliance(batch, outputs)
                self.compliance_history.append(compliance_result)
                
                # Update privacy parameters
                await self.privacy.adapt_privacy({
                    "loss": loss.item(),
                    "compliance_score": compliance_result["compliance_score"]
                })
            
            # Epoch end compliance check
            epoch_compliance = await self._check_epoch_compliance(epoch_loss)
            self.training_history.append({
                "epoch": epoch,
                "loss": epoch_loss,
                "compliance": epoch_compliance
            })
            
            # Validate if validation data provided
            if val_dataloader:
                val_metrics = await self._validate(val_dataloader)
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"Training Loss: {epoch_loss:.4f}")
                print(f"Validation Metrics: {val_metrics}")
                print(f"Compliance Score: {epoch_compliance['compliance_score']:.4f}")
    
    async def _private_forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        """Perform forward pass with privacy protection."""
        # Apply differential privacy to inputs
        private_inputs = self.privacy.dp_mechanism.privatize(batch)
        
        # Forward pass
        outputs = self.model(**private_inputs)
        
        return outputs
    
    async def _check_compliance(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Any
    ) -> Dict[str, Any]:
        """Check compliance for a batch."""
        # Prepare compliance data
        compliance_data = {
            "inputs": batch,
            "outputs": outputs,
            "model_state": self.model.state_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Verify compliance
        is_compliant, result = await self.compliance_shard.verify_compliance(
            compliance_data,
            level=ComplianceLevel.ADVANCED
        )
        
        return {
            "is_compliant": is_compliant,
            "compliance_score": result["metrics"].score,
            "proof": result["proof"],
            "private_result": result["private_result"]
        }
    
    async def _check_epoch_compliance(self, epoch_loss: float) -> Dict[str, Any]:
        """Check compliance at the end of an epoch."""
        # Generate explanation for training progress
        explanation = await self.explainer.explain_decision({
            "epoch_loss": epoch_loss,
            "model_state": self.model.state_dict(),
            "compliance_history": self.compliance_history[-100:]  # Last 100 checks
        })
        
        return {
            "compliance_score": explanation["confidence"],
            "explanation": explanation,
            "metrics": self.compliance_shard.metrics_history[-1]
        }
    
    async def _validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Validate model with compliance checks."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                outputs = await self._private_forward(batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                predictions = outputs.logits.argmax(dim=-1)
                correct += (predictions == batch["labels"]).sum().item()
                total += len(batch["labels"])
        
        return {
            "loss": total_loss / len(val_dataloader),
            "accuracy": correct / total
        }
    
    def _prepare_dataloader(
        self,
        data: List[Dict[str, Any]],
        batch_size: int
    ) -> DataLoader:
        """Prepare dataloader with tokenization."""
        # Tokenize data
        encodings = self.tokenizer(
            [item["text"] for item in data],
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            torch.tensor([item["label"] for item in data])
        )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    async def save_model(self, path: str):
        """Save model with compliance proofs and watermarks."""
        # Apply watermark
        watermarked_model = await self.watermarking.watermark_model(self.model)
        
        # Generate final compliance proof
        final_compliance = await self._check_epoch_compliance(0.0)  # Use 0.0 as dummy loss
        
        # Save model and metadata
        torch.save({
            "model_state": watermarked_model.state_dict(),
            "compliance_proof": final_compliance,
            "training_history": self.training_history,
            "compliance_history": self.compliance_history,
            "watermark_info": await self.watermarking.track_fingerprint(watermarked_model)
        }, path)

async def main():
    """Example usage of CompliantModelTrainer."""
    # Initialize trainer
    trainer = CompliantModelTrainer(
        model_name="bert-base-uncased",
        compliance_config={
            "epsilon": 1.0,
            "rules": [
                {"name": "privacy", "threshold": 0.8},
                {"name": "fairness", "threshold": 0.9},
                {"name": "transparency", "threshold": 0.7}
            ]
        }
    )
    
    # Example training data
    train_data = [
        {"text": "This is a positive example", "label": 1},
        {"text": "This is a negative example", "label": 0},
        # Add more examples...
    ]
    
    # Finetune model
    await trainer.finetune(
        train_data=train_data,
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5
    )
    
    # Save model
    await trainer.save_model("compliant_model.pt")

if __name__ == "__main__":
    asyncio.run(main()) 