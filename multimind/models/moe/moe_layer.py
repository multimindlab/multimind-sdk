# Optional torch import for MoE features
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. MoE features will be disabled.")

from typing import Optional, Tuple, List
import math

if TORCH_AVAILABLE:
    class MoELayer(nn.Module):
        """
        Mixture of Experts (MoE) layer implementation with advanced routing and load balancing.
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
            use_noisy_gate: bool = True
        ):
            super().__init__()
            self.input_dim = input_dim
            self.num_experts = num_experts
            self.expert_dim = expert_dim
            self.k = k
            self.capacity_factor = capacity_factor
            self.use_aux_loss = use_aux_loss
            self.use_noisy_gate = use_noisy_gate

            # Expert networks
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, expert_dim),
                    nn.LayerNorm(expert_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_dim, input_dim)
                ) for _ in range(num_experts)
            ])

            # Router network with noise
            self.router = nn.Linear(input_dim, num_experts)
            if use_noisy_gate:
                self.noise_epsilon = 1e-2
                self.register_buffer("noise_weights", torch.randn(num_experts))

            # Load balancing metrics
            self.register_buffer("expert_usage", torch.zeros(num_experts))
            self.register_buffer("expert_loss", torch.zeros(num_experts))

        def _noisy_gate(self, logits: torch.Tensor) -> torch.Tensor:
            """Add noise to the gate logits for better exploration."""
            if self.use_noisy_gate and self.training:
                noise = torch.randn_like(logits) * self.noise_epsilon
                return logits + noise
            return logits

        def _load_balancing_loss(self, router_probs: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
            """Calculate load balancing loss to ensure even expert utilization."""
            if not self.use_aux_loss:
                return torch.tensor(0.0, device=router_probs.device)

            # Calculate expert utilization
            expert_usage = torch.zeros(self.num_experts, device=router_probs.device)
            for i in range(self.k):
                expert_usage.scatter_add_(0, expert_indices[:, i], router_probs[:, i])

            # Calculate load balancing loss
            mean_expert_usage = expert_usage.mean()
            load_balancing_loss = torch.sum(expert_usage * mean_expert_usage) / self.num_experts
            return load_balancing_loss

        def _capacity_loss(self, router_probs: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
            """Calculate capacity loss to prevent overloading experts."""
            if not self.use_aux_loss:
                return torch.tensor(0.0, device=router_probs.device)

            # Calculate capacity constraints
            capacity = math.ceil(router_probs.size(0) * self.capacity_factor / self.num_experts)
            expert_counts = torch.zeros(self.num_experts, device=router_probs.device)
            for i in range(self.k):
                expert_counts.scatter_add_(0, expert_indices[:, i], torch.ones_like(expert_indices[:, i], dtype=torch.float))

            # Calculate capacity loss
            capacity_loss = torch.sum(torch.relu(expert_counts - capacity)) / router_probs.size(0)
            return capacity_loss

        def forward(
            self,
            x: torch.Tensor,
            return_aux_loss: bool = False
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """
            Forward pass through the MoE layer.
            
            Args:
                x: Input tensor of shape [batch_size, seq_len, input_dim]
                return_aux_loss: Whether to return auxiliary losses
                
            Returns:
                Tuple of (output tensor, auxiliary loss if requested)
            """
            batch_size, seq_len, _ = x.shape
            x_reshaped = x.view(-1, self.input_dim)

            # Get routing weights with noise
            router_logits = self.router(x_reshaped)
            router_logits = self._noisy_gate(router_logits)
            router_probs = F.softmax(router_logits, dim=-1)

            # Select top-k experts
            top_k_weights, top_k_indices = torch.topk(router_probs, self.k)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

            # Apply experts
            expert_outputs = []
            for i in range(self.k):
                expert_idx = top_k_indices[:, i]
                expert_output = torch.stack([
                    self.experts[idx](x_reshaped[j]) for j, idx in enumerate(expert_idx)
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

            return output, aux_loss

        def get_expert_usage(self) -> torch.Tensor:
            """Get the current expert usage statistics."""
            return self.expert_usage / (self.expert_usage.sum() + 1e-6)

        def reset_expert_usage(self):
            """Reset expert usage statistics."""
            self.expert_usage.zero_()
            self.expert_loss.zero_()
else:
    class MoELayer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for MoELayer. Please install torch.") 