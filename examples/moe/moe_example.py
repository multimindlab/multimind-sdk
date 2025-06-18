import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from multimind.models.moe.moe_model import MoEModel
from multimind.fine_tuning.moe_tuning import MoETrainer
from multimind.config.moe_config import MoEConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_data(num_samples: int, input_dim: int, num_classes: int) -> tuple:
    """Create dummy data for demonstration."""
    # Generate random input data
    X = torch.randn(num_samples, input_dim)
    
    # Generate random labels
    y = torch.randint(0, num_classes, (num_samples,))
    
    return X, y

def main():
    # Create configuration
    config = MoEConfig(
        input_dim=768,
        hidden_dim=1024,
        num_experts=8,
        num_layers=6,
        num_heads=8,
        k=2,
        batch_size=32,
        num_epochs=5
    )
    config.validate()

    # Create dummy dataset
    num_samples = 1000
    num_classes = 10
    X_train, y_train = create_dummy_data(num_samples, config.input_dim, num_classes)
    X_val, y_val = create_dummy_data(200, config.input_dim, num_classes)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # Create model
    model = MoEModel(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_experts=config.num_experts,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        k=config.k,
        dropout=config.dropout,
        expert_dropout=config.expert_dropout,
        use_aux_loss=config.use_aux_loss,
        use_noisy_gate=config.use_noisy_gate
    )

    # Create trainer
    trainer = MoETrainer(
        model=model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        aux_loss_weight=config.aux_loss_weight,
        expert_balance_weight=config.expert_balance_weight,
        device=config.device
    )

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Train model
    logger.info("Starting training...")
    metrics = trainer.train(
        train_loader=train_loader,
        task_loss_fn=criterion,
        num_epochs=config.num_epochs,
        eval_loader=val_loader,
        eval_loss_fn=criterion,
        checkpoint_path="checkpoints/moe_model.pt" if config.checkpoint_dir else None
    )

    # Print final metrics
    logger.info("Training completed!")
    logger.info(f"Final training loss: {metrics['train_loss'][-1]:.4f}")
    if metrics['eval_loss']:
        logger.info(f"Final evaluation loss: {metrics['eval_loss'][-1]:.4f}")

    # Example inference
    logger.info("\nPerforming inference on a sample...")
    model.eval()
    with torch.no_grad():
        sample_input = torch.randn(1, config.input_dim).to(config.device)
        output, _ = model(sample_input)
        logger.info(f"Output shape: {output.shape}")

    # Print expert usage statistics
    expert_usage = model.get_expert_usage()
    logger.info("\nExpert usage statistics:")
    for layer_idx, usage in expert_usage.items():
        logger.info(f"Layer {layer_idx}:")
        for expert_idx, usage_value in enumerate(usage):
            logger.info(f"  Expert {expert_idx}: {usage_value:.4f}")

if __name__ == "__main__":
    main() 