import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, List, Tuple
from torch.utils.data import DataLoader
from ..models.moe.moe_model import MoEModel
import logging
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

class MoETrainer:
    """
    Trainer for fine-tuning MoE models with advanced strategies.
    """
    def __init__(
        self,
        model: MoEModel,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        aux_loss_weight: float = 0.01,
        expert_balance_weight: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.aux_loss_weight = aux_loss_weight
        self.expert_balance_weight = expert_balance_weight
        self.max_grad_norm = max_grad_norm

        # Initialize optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=warmup_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        # Initialize metrics
        self.metrics = {
            'train_loss': [],
            'aux_loss': [],
            'expert_usage': []
        }

    def _calculate_expert_balance_loss(self) -> torch.Tensor:
        """Calculate loss to encourage balanced expert usage."""
        expert_usage = self.model.get_expert_usage()
        balance_loss = 0.0
        
        for layer_usage in expert_usage.values():
            # Calculate variance of expert usage
            mean_usage = layer_usage.mean()
            variance = torch.mean((layer_usage - mean_usage) ** 2)
            balance_loss += variance
            
        return balance_loss / len(expert_usage)

    def train_step(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        task_loss_fn: nn.Module
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Input batch tensor
            labels: Target labels
            task_loss_fn: Loss function for the main task
            
        Returns:
            Dictionary of metrics for this step
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs, aux_losses = self.model(batch, return_aux_loss=True)
        
        # Calculate task loss
        task_loss = task_loss_fn(outputs, labels)
        
        # Calculate auxiliary losses
        aux_loss = aux_losses['total_aux_loss'] if aux_losses else 0.0
        balance_loss = self._calculate_expert_balance_loss()
        
        # Combine losses
        total_loss = (
            task_loss +
            self.aux_loss_weight * aux_loss +
            self.expert_balance_weight * balance_loss
        )

        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        # Update metrics
        metrics = {
            'task_loss': task_loss.item(),
            'aux_loss': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
            'balance_loss': balance_loss.item(),
            'total_loss': total_loss.item()
        }

        return metrics

    def train_epoch(
        self,
        train_loader: DataLoader,
        task_loss_fn: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            task_loss_fn: Loss function for the main task
            epoch: Current epoch number
            
        Returns:
            Dictionary of average metrics for the epoch
        """
        epoch_metrics = {
            'task_loss': [],
            'aux_loss': [],
            'balance_loss': [],
            'total_loss': []
        }

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch, labels in progress_bar:
            batch = batch.to(self.device)
            labels = labels.to(self.device)
            
            # Training step
            step_metrics = self.train_step(batch, labels, task_loss_fn)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{step_metrics['total_loss']:.4f}",
                'task_loss': f"{step_metrics['task_loss']:.4f}"
            })
            
            # Update metrics
            for key, value in step_metrics.items():
                epoch_metrics[key].append(value)

        # Calculate average metrics
        avg_metrics = {
            key: np.mean(values)
            for key, values in epoch_metrics.items()
        }

        # Log metrics
        logger.info(f"Epoch {epoch} metrics:")
        for key, value in avg_metrics.items():
            logger.info(f"{key}: {value:.4f}")

        return avg_metrics

    def train(
        self,
        train_loader: DataLoader,
        task_loss_fn: nn.Module,
        num_epochs: int,
        eval_loader: Optional[DataLoader] = None,
        eval_loss_fn: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            task_loss_fn: Loss function for the main task
            num_epochs: Number of epochs to train
            eval_loader: Optional evaluation data loader
            eval_loss_fn: Optional evaluation loss function
            checkpoint_path: Optional path to save checkpoints
            
        Returns:
            Dictionary of training metrics
        """
        best_eval_loss = float('inf')
        training_metrics = {
            'train_loss': [],
            'eval_loss': [] if eval_loader else None
        }

        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, task_loss_fn, epoch)
            training_metrics['train_loss'].append(train_metrics['total_loss'])

            # Evaluation
            if eval_loader and eval_loss_fn:
                eval_metrics = self.evaluate(eval_loader, eval_loss_fn)
                training_metrics['eval_loss'].append(eval_metrics['loss'])

                # Save best model
                if eval_metrics['loss'] < best_eval_loss and checkpoint_path:
                    best_eval_loss = eval_metrics['loss']
                    self.save_checkpoint(checkpoint_path)

        return training_metrics

    def evaluate(
        self,
        eval_loader: DataLoader,
        loss_fn: nn.Module
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            eval_loader: Evaluation data loader
            loss_fn: Loss function for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch, labels in eval_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                
                outputs, _ = self.model(batch, return_aux_loss=False)
                loss = loss_fn(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.metrics = checkpoint['metrics'] 