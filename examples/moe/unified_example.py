import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from multimind.models.moe.unified_moe import UnifiedMoE
from multimind.models.moe.advanced_moe import AdvancedMoELayer
from multimind.models.moe.moe import Expert, TextExpert, VisionExpert
from multimind.config.moe_config import MoEConfig
import logging
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyTextExpert(TextExpert):
    """Dummy text expert for demonstration."""
    async def process(self, input_data: str) -> dict:
        # Simulate text processing
        return {
            "embedding": torch.randn(768),
            "confidence": 0.9
        }

class DummyVisionExpert(VisionExpert):
    """Dummy vision expert for demonstration."""
    async def process(self, input_data: Any) -> dict:
        # Simulate image processing
        return {
            "embedding": torch.randn(768),
            "confidence": 0.85
        }

def create_dummy_data(num_samples: int, input_dim: int, num_classes: int) -> tuple:
    """Create dummy data for demonstration."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y

async def run_neural_moe_example():
    """Run example with neural MoE implementation."""
    logger.info("\nRunning Neural MoE Example:")
    
    # Create configuration
    config = {
        'input_dim': 768,
        'hidden_dim': 1024,
        'num_experts': 8,
        'num_layers': 6,
        'num_heads': 8,
        'k': 2,
        'batch_size': 32,
        'num_epochs': 5,
        'use_gradient_checkpointing': True,
        'expert_specialization': True
    }
    
    # Create model
    model = UnifiedMoE(mode="neural", config=config)
    
    # Create dummy data
    X_train, y_train = create_dummy_data(1000, config['input_dim'], 10)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    # Training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        
        for batch, labels in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            result = await model.process(batch, return_aux_loss=True)
            output = result['output']
            aux_loss = result['aux_loss']
            
            # Calculate loss
            task_loss = criterion(output, labels)
            loss = task_loss + 0.01 * aux_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        
        # Print expert statistics
        stats = model.get_expert_info()
        logger.info(f"Expert usage: {stats['usage']}")

async def run_modality_moe_example():
    """Run example with modality-based MoE implementation."""
    logger.info("\nRunning Modality-based MoE Example:")
    
    # Create experts
    experts = {
        'text': DummyTextExpert(None),
        'vision': DummyVisionExpert(None)
    }
    
    # Create model
    model = UnifiedMoE(
        mode="modality",
        config={'hidden_size': 768},
        experts=experts
    )
    
    # Example multi-modal input
    input_data = {
        'text': "Sample text input",
        'vision': torch.randn(3, 224, 224)  # Dummy image
    }
    
    # Process input
    result = await model.process(input_data)
    
    # Print results
    logger.info("Processing results:")
    logger.info(f"Output shape: {result['output'].shape}")
    logger.info(f"Expert weights: {result['expert_weights']}")
    
    # Print expert information
    expert_info = model.get_expert_info()
    logger.info(f"Available experts: {expert_info['experts']}")
    logger.info(f"Modalities: {expert_info['modalities']}")

async def main():
    """Run both examples."""
    # Run neural MoE example
    await run_neural_moe_example()
    
    # Run modality-based MoE example
    await run_modality_moe_example()
    
    # Demonstrate expert management
    logger.info("\nDemonstrating Expert Management:")
    
    # Create modality-based model
    experts = {
        'text': DummyTextExpert(None),
        'vision': DummyVisionExpert(None)
    }
    model = UnifiedMoE(mode="modality", experts=experts)
    
    # Add new expert
    model.add_expert('audio', DummyVisionExpert(None))
    logger.info("Added new expert")
    
    # Get updated expert info
    expert_info = model.get_expert_info()
    logger.info(f"Updated experts: {expert_info['experts']}")
    
    # Remove expert
    model.remove_expert('audio')
    logger.info("Removed audio expert")
    
    # Get final expert info
    expert_info = model.get_expert_info()
    logger.info(f"Final experts: {expert_info['experts']}")

if __name__ == "__main__":
    asyncio.run(main()) 