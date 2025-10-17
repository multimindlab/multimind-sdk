# Optional torch import for MoE model features
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. MoE model features will be disabled.")

from typing import Optional, Dict, Any, Tuple
from .moe_layer import MoELayer

if TORCH_AVAILABLE:
    class MoEModel(nn.Module):
        """
        Main Mixture of Experts model implementation.
        """
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_experts: int,
            num_layers: int,
            num_heads: int = 8,
            dropout: float = 0.1,
            expert_dropout: float = 0.1,
            k: int = 2,
            capacity_factor: float = 1.0,
            use_aux_loss: bool = True,
            use_noisy_gate: bool = True
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_experts = num_experts
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.k = k

            # Input projection
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.input_norm = nn.LayerNorm(hidden_dim)

            # MoE layers
            self.moe_layers = nn.ModuleList([
                MoELayer(
                    input_dim=hidden_dim,
                    num_experts=num_experts,
                    expert_dim=hidden_dim * 4,  # FFN expansion factor
                    k=k,
                    capacity_factor=capacity_factor,
                    dropout=expert_dropout,
                    use_aux_loss=use_aux_loss,
                    use_noisy_gate=use_noisy_gate
                ) for _ in range(num_layers)
            ])

            # Layer norms
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])

            # Output projection
            self.output_proj = nn.Linear(hidden_dim, input_dim)
            self.output_norm = nn.LayerNorm(input_dim)

            # Dropout
            self.dropout = nn.Dropout(dropout)

            # Initialize weights
            self._init_weights()

        def _init_weights(self):
            """Initialize model weights."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

        def forward(
            self,
            x: torch.Tensor,
            return_aux_loss: bool = False
        ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
            """
            Forward pass through the MoE model.
            
            Args:
                x: Input tensor of shape [batch_size, seq_len, input_dim]
                return_aux_loss: Whether to return auxiliary losses
                
            Returns:
                Tuple of (output tensor, auxiliary losses if requested)
            """
            # Input projection
            x = self.input_proj(x)
            x = self.input_norm(x)
            x = self.dropout(x)

            # Track auxiliary losses
            aux_losses = {}
            total_aux_loss = 0.0

            # Process through MoE layers
            for i, (moe_layer, layer_norm) in enumerate(zip(self.moe_layers, self.layer_norms)):
                # Layer norm
                x = layer_norm(x)
                
                # MoE layer
                moe_output, aux_loss = moe_layer(x, return_aux_loss=return_aux_loss)
                x = x + moe_output  # Residual connection
                
                if return_aux_loss and aux_loss is not None:
                    aux_losses[f"layer_{i}_aux_loss"] = aux_loss
                    total_aux_loss += aux_loss

            # Output projection
            x = self.output_proj(x)
            x = self.output_norm(x)

            if return_aux_loss:
                aux_losses["total_aux_loss"] = total_aux_loss
                return x, aux_losses
            else:
                return x, None

        def get_expert_usage(self) -> Dict[int, torch.Tensor]:
            """Get expert usage statistics for all layers."""
            usage_stats = {}
            for i, moe_layer in enumerate(self.moe_layers):
                usage_stats[i] = moe_layer.get_expert_usage()
            return usage_stats

        def reset_expert_usage(self):
            """Reset expert usage statistics for all layers."""
            for moe_layer in self.moe_layers:
                moe_layer.reset_expert_usage()

        def get_config(self) -> Dict[str, Any]:
            """Get model configuration."""
            return {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_experts": self.num_experts,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "k": self.k
            }

else:
    class MoEModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for MoEModel. Please install torch.") 