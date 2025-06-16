from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import os

@dataclass
class MoEConfig:
    """Configuration for MoE model and training."""
    # Model architecture
    input_dim: int
    hidden_dim: int
    num_experts: int
    num_layers: int
    num_heads: int = 8
    k: int = 2  # Number of experts to use per token
    capacity_factor: float = 1.0
    dropout: float = 0.1
    expert_dropout: float = 0.1
    use_aux_loss: bool = True
    use_noisy_gate: bool = True

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    aux_loss_weight: float = 0.01
    expert_balance_weight: float = 0.1
    batch_size: int = 32
    num_epochs: int = 10

    # Optional settings
    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.input_dim > 0, "input_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_experts > 0, "num_experts must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.num_heads > 0, "num_heads must be positive"
        assert 0 < self.k <= self.num_experts, "k must be between 1 and num_experts"
        assert 0 < self.capacity_factor <= 2.0, "capacity_factor must be between 0 and 2"
        assert 0 <= self.dropout < 1, "dropout must be between 0 and 1"
        assert 0 <= self.expert_dropout < 1, "expert_dropout must be between 0 and 1"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert self.warmup_steps > 0, "warmup_steps must be positive"
        assert self.max_grad_norm > 0, "max_grad_norm must be positive"
        assert self.aux_loss_weight >= 0, "aux_loss_weight must be non-negative"
        assert self.expert_balance_weight >= 0, "expert_balance_weight must be non-negative"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_experts': self.num_experts,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'k': self.k,
                'capacity_factor': self.capacity_factor,
                'dropout': self.dropout,
                'expert_dropout': self.expert_dropout,
                'use_aux_loss': self.use_aux_loss,
                'use_noisy_gate': self.use_noisy_gate
            },
            'training': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'warmup_steps': self.warmup_steps,
                'max_grad_norm': self.max_grad_norm,
                'aux_loss_weight': self.aux_loss_weight,
                'expert_balance_weight': self.expert_balance_weight,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs
            },
            'paths': {
                'checkpoint_dir': self.checkpoint_dir,
                'log_dir': self.log_dir
            },
            'device': self.device
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MoEConfig':
        """Create configuration from dictionary."""
        model_config = config_dict.get('model', {})
        training_config = config_dict.get('training', {})
        paths_config = config_dict.get('paths', {})

        return cls(
            # Model parameters
            input_dim=model_config.get('input_dim', 768),
            hidden_dim=model_config.get('hidden_dim', 1024),
            num_experts=model_config.get('num_experts', 8),
            num_layers=model_config.get('num_layers', 6),
            num_heads=model_config.get('num_heads', 8),
            k=model_config.get('k', 2),
            capacity_factor=model_config.get('capacity_factor', 1.0),
            dropout=model_config.get('dropout', 0.1),
            expert_dropout=model_config.get('expert_dropout', 0.1),
            use_aux_loss=model_config.get('use_aux_loss', True),
            use_noisy_gate=model_config.get('use_noisy_gate', True),

            # Training parameters
            learning_rate=training_config.get('learning_rate', 1e-4),
            weight_decay=training_config.get('weight_decay', 0.01),
            warmup_steps=training_config.get('warmup_steps', 1000),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            aux_loss_weight=training_config.get('aux_loss_weight', 0.01),
            expert_balance_weight=training_config.get('expert_balance_weight', 0.1),
            batch_size=training_config.get('batch_size', 32),
            num_epochs=training_config.get('num_epochs', 10),

            # Paths
            checkpoint_dir=paths_config.get('checkpoint_dir'),
            log_dir=paths_config.get('log_dir'),
            device=config_dict.get('device', "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
        )

    def save(self, path: str) -> None:
        """Save configuration to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'MoEConfig':
        """Load configuration from file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def get_default_config(cls) -> 'MoEConfig':
        """Get default configuration."""
        return cls(
            input_dim=768,
            hidden_dim=1024,
            num_experts=8,
            num_layers=6
        ) 