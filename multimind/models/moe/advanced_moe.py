# Optional torch import for advanced MoE features
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Advanced MoE features will be disabled.")

from typing import Dict, List, Any, Optional, Tuple
import math
from .moe_layer import MoELayer

if TORCH_AVAILABLE:
    class AdvancedMoELayer(MoELayer):
        """
        Advanced MoE layer with additional features:
        - Dynamic expert capacity
        - Expert specialization
        - Adaptive routing
        - Expert pruning
        - Gradient checkpointing
        """
        def __init__(
            self,
            input_dim: int,
            num_experts: int,
            expert_dim: int,
            k: int = 2,
            capacity_factor: float = 1.0,
            dropout: float = 0.1,
            use_aux_loss: bool = True,
            use_noisy_gate: bool = True,
            use_gradient_checkpointing: bool = False,
            expert_specialization: bool = False,
            min_expert_capacity: int = 4,
            max_expert_capacity: int = 256,
            pruning_threshold: float = 0.1
        ):
            super().__init__(
                input_dim=input_dim,
                num_experts=num_experts,
                expert_dim=expert_dim,
                k=k,
                capacity_factor=capacity_factor,
                dropout=dropout,
                use_aux_loss=use_aux_loss,
                use_noisy_gate=use_noisy_gate
            )
            
            self.use_gradient_checkpointing = use_gradient_checkpointing
            self.expert_specialization = expert_specialization
            self.min_expert_capacity = min_expert_capacity
            self.max_expert_capacity = max_expert_capacity
            self.pruning_threshold = pruning_threshold
            
            # Expert specialization parameters
            if expert_specialization:
                self.expert_embeddings = nn.Parameter(
                    torch.randn(num_experts, input_dim)
                )
                self.specialization_router = nn.Linear(input_dim, num_experts)
            
            # Expert pruning parameters
            self.register_buffer("expert_importance", torch.ones(num_experts))
            self.register_buffer("expert_usage_count", torch.zeros(num_experts))
            
            # Dynamic capacity parameters
            self.register_buffer("current_capacity", torch.ones(num_experts) * min_expert_capacity)

        def _compute_dynamic_capacity(self, batch_size: int) -> torch.Tensor:
            """Compute dynamic capacity for each expert based on usage."""
            if not self.training:
                return self.current_capacity
            
            # Update capacity based on expert usage
            usage_ratio = self.expert_usage / (self.expert_usage.sum() + 1e-6)
            target_capacity = torch.clamp(
                usage_ratio * batch_size,
                min=self.min_expert_capacity,
                max=self.max_expert_capacity
            )
            
            # Smooth capacity updates
            self.current_capacity = (
                0.9 * self.current_capacity +
                0.1 * target_capacity
            )
            
            return self.current_capacity

        def _compute_specialization_weights(
            self,
            x: torch.Tensor
        ) -> torch.Tensor:
            """Compute expert specialization weights."""
            if not self.expert_specialization:
                return None
            
            # Compute similarity between input and expert embeddings
            similarity = F.cosine_similarity(
                x.unsqueeze(1),
                self.expert_embeddings.unsqueeze(0),
                dim=-1
            )
            
            # Combine with router weights
            router_weights = self.specialization_router(x)
            combined_weights = F.softmax(
                similarity + router_weights,
                dim=-1
            )
            
            return combined_weights

        def _prune_experts(self) -> None:
            """Prune experts based on importance and usage."""
            if not self.training:
                return
            
            # Update expert importance
            self.expert_importance = (
                0.9 * self.expert_importance +
                0.1 * (self.expert_usage / (self.expert_usage.sum() + 1e-6))
            )
            
            # Mark experts for pruning
            prune_mask = self.expert_importance < self.pruning_threshold
            if prune_mask.any():
                logger.info(f"Pruning {prune_mask.sum()} experts")
                self.expert_importance[prune_mask] = 0.0

        def _apply_gradient_checkpointing(
            self,
            expert_idx: int,
            x: torch.Tensor
        ) -> torch.Tensor:
            """Apply gradient checkpointing to expert computation."""
            if not self.use_gradient_checkpointing:
                return self.experts[expert_idx](x)
            
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(inputs[0])
                return custom_forward
            
            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.experts[expert_idx]),
                x
            )

        def forward(
            self,
            x: torch.Tensor,
            return_aux_loss: bool = False
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """
            Forward pass with advanced features.
            """
            batch_size, seq_len, _ = x.shape
            x_reshaped = x.view(-1, self.input_dim)
            
            # Compute dynamic capacity
            capacity = self._compute_dynamic_capacity(batch_size * seq_len)
            
            # Get routing weights with specialization
            router_logits = self.router(x_reshaped)
            router_logits = self._noisy_gate(router_logits)
            
            if self.expert_specialization:
                spec_weights = self._compute_specialization_weights(x_reshaped)
                if spec_weights is not None:
                    router_logits = router_logits + spec_weights
            
            router_probs = F.softmax(router_logits, dim=-1)
            
            # Select top-k experts with capacity constraints
            top_k_weights, top_k_indices = torch.topk(router_probs, self.k)
            
            # Apply capacity constraints
            expert_counts = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.k):
                expert_counts.scatter_add_(0, top_k_indices[:, i], torch.ones_like(top_k_indices[:, i], dtype=torch.float))
            
            # Filter out experts that exceed capacity
            capacity_mask = expert_counts <= capacity
            valid_experts = torch.where(capacity_mask)[0]
            
            if len(valid_experts) == 0:
                # Fallback to basic routing
                return super().forward(x, return_aux_loss)
            
            # Apply experts with gradient checkpointing
            expert_outputs = []
            for i in range(self.k):
                expert_idx = top_k_indices[:, i]
                expert_output = torch.stack([
                    self._apply_gradient_checkpointing(idx, x_reshaped[j]) 
                    for j, idx in enumerate(expert_idx)
                ])
                expert_outputs.append(expert_output * top_k_weights[:, i].unsqueeze(-1))
            
            # Combine expert outputs
            output = sum(expert_outputs)
            output = output.view(batch_size, seq_len, self.input_dim)
            
            # Calculate auxiliary losses if requested
            aux_loss = None
            if return_aux_loss and self.use_aux_loss:
                load_balancing_loss = self._load_balancing_loss(router_probs, top_k_indices)
                capacity_loss = self._capacity_loss(router_probs, top_k_indices)
                aux_loss = load_balancing_loss + capacity_loss
            
            # Update expert usage statistics
            if self.training:
                with torch.no_grad():
                    for i in range(self.k):
                        self.expert_usage.scatter_add_(0, top_k_indices[:, i], top_k_weights[:, i])
                        self.expert_usage_count.scatter_add_(0, top_k_indices[:, i], torch.ones_like(top_k_indices[:, i], dtype=torch.float))
            
            # Prune experts if needed
            self._prune_experts()
            
            return output, aux_loss

        def get_expert_stats(self) -> Dict[str, Any]:
            """Get comprehensive expert statistics."""
            return {
                "expert_usage": self.expert_usage.tolist(),
                "expert_importance": self.expert_importance.tolist(),
                "expert_usage_count": self.expert_usage_count.tolist(),
                "current_capacity": self.current_capacity.tolist(),
                "total_experts": self.num_experts,
                "active_experts": (self.expert_importance > 0).sum().item()
            }

    class MoEFactory:
        """Factory class for creating MoE models and components."""
        
        @staticmethod
        def create_moe_model(
            model_type: str = "unified",
            config: Optional[Dict[str, Any]] = None,
            **kwargs
        ):
            """Create a MoE model based on type."""
            if config is None:
                config = {}
            
            if model_type == "unified":
                return AdvancedMoELayer(**{**config, **kwargs})
            elif model_type == "basic":
                return MoELayer(**{**config, **kwargs})
            else:
                raise ValueError(f"Unknown MoE model type: {model_type}")
        
        @staticmethod
        def create_expert(
            expert_type: str,
            expert_id: str,
            **kwargs
        ):
            """Create an expert component."""
            if expert_type == "mlp":
                input_dim = kwargs.get("input_dim", 512)
                hidden_dim = kwargs.get("hidden_dim", 1024)
                return nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(kwargs.get("dropout", 0.1)),
                    nn.Linear(hidden_dim, input_dim)
                )
            elif expert_type == "transformer":
                # Transformer expert implementation
                pass
            else:
                raise ValueError(f"Unknown expert type: {expert_type}")
        
        @staticmethod
        def create_router(
            router_type: str,
            experts: Dict[str, Any],
            **kwargs
        ):
            """Create a router component."""
            if router_type == "linear":
                input_dim = kwargs.get("input_dim", 512)
                num_experts = len(experts)
                return nn.Linear(input_dim, num_experts)
            elif router_type == "attention":
                # Attention-based router implementation
                pass
            else:
                raise ValueError(f"Unknown router type: {router_type}")

else:
    class AdvancedMoELayer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for AdvancedMoELayer. Please install torch.")
    
    class MoEFactory:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for MoEFactory. Please install torch.") 