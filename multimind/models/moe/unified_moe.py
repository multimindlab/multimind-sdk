from typing import Dict, List, Any, Optional, Union, Type
# Optional torch import for unified MoE features
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Unified MoE features will be disabled.")

from abc import ABC, abstractmethod
from .moe_layer import MoELayer
from .moe_model import MoEModel
from .moe import Expert, MoEBase, ExpertRouter
from ..base import BaseLLM
import logging

logger = logging.getLogger(__name__)

if TORCH_AVAILABLE:
    class UnifiedMoE(nn.Module):
        """
        Unified interface for both neural and modality-based MoE implementations.
        """
        def __init__(
            self,
            mode: str = "neural",  # "neural" or "modality"
            config: Dict[str, Any] = None,
            experts: Optional[Dict[str, Expert]] = None
        ):
            super().__init__()
            self.mode = mode
            self.config = config or {}
            
            if mode == "neural":
                self.model = self._create_neural_moe()
            else:
                self.model = self._create_modality_moe(experts)
                
            # Initialize metrics
            self.metrics = {
                'expert_usage': {},
                'routing_weights': {},
                'performance_metrics': {}
            }

        def _create_neural_moe(self) -> MoEModel:
            """Create neural network-based MoE model."""
            return MoEModel(
                input_dim=self.config.get('input_dim', 768),
                hidden_dim=self.config.get('hidden_dim', 1024),
                num_experts=self.config.get('num_experts', 8),
                num_layers=self.config.get('num_layers', 6),
                num_heads=self.config.get('num_heads', 8),
                k=self.config.get('k', 2),
                dropout=self.config.get('dropout', 0.1),
                expert_dropout=self.config.get('expert_dropout', 0.1),
                use_aux_loss=self.config.get('use_aux_loss', True),
                use_noisy_gate=self.config.get('use_noisy_gate', True)
            )

        def _create_modality_moe(self, experts: Optional[Dict[str, Expert]]) -> MoEBase:
            """Create modality-based MoE model."""
            if not experts:
                raise ValueError("Experts must be provided for modality-based MoE")
            return MoEBase(
                experts=experts,
                hidden_size=self.config.get('hidden_size', 768),
                num_experts=len(experts)
            )

        async def process(
            self,
            input_data: Union[torch.Tensor, Dict[str, Any]],
            return_aux_loss: bool = False
        ) -> Dict[str, Any]:
            """
            Process input through the MoE model.
            
            Args:
                input_data: Input tensor or dictionary of modality inputs
                return_aux_loss: Whether to return auxiliary losses
                
            Returns:
                Dictionary containing model outputs and optional metrics
            """
            if self.mode == "neural":
                return await self._process_neural(input_data, return_aux_loss)
            else:
                return await self._process_modality(input_data)

        async def _process_neural(
            self,
            input_data: torch.Tensor,
            return_aux_loss: bool
        ) -> Dict[str, Any]:
            """Process input through neural MoE model."""
            output, aux_loss = self.model(input_data, return_aux_loss)
            
            # Update metrics
            self._update_neural_metrics()
            
            return {
                'output': output,
                'aux_loss': aux_loss,
                'metrics': self.metrics
            }

        async def _process_modality(
            self,
            input_data: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Process input through modality-based MoE model."""
            result = await self.model.process(input_data)
            
            # Update metrics
            self._update_modality_metrics(result)
            
            return result

        def _update_neural_metrics(self):
            """Update neural MoE metrics."""
            if hasattr(self.model, 'get_expert_usage'):
                self.metrics['expert_usage'] = self.model.get_expert_usage()

        def _update_modality_metrics(self, result: Dict[str, Any]):
            """Update modality MoE metrics."""
            if 'expert_usage' in result:
                self.metrics['expert_usage'] = result['expert_usage']

        def get_metrics(self) -> Dict[str, Any]:
            """Get current metrics."""
            return self.metrics

        def reset_metrics(self):
            """Reset all metrics."""
            self.metrics = {
                'expert_usage': {},
                'routing_weights': {},
                'performance_metrics': {}
            }

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass for neural MoE."""
            if self.mode == "neural":
                output, _ = self.model(x, return_aux_loss=False)
                return output
            else:
                raise ValueError("Forward pass only available for neural MoE")

        def save_checkpoint(self, path: str):
            """Save model checkpoint."""
            torch.save({
                'model_state_dict': self.state_dict(),
                'config': self.config,
                'mode': self.mode,
                'metrics': self.metrics
            }, path)

        @classmethod
        def load_checkpoint(cls, path: str) -> 'UnifiedMoE':
            """Load model from checkpoint."""
            checkpoint = torch.load(path)
            model = cls(
                mode=checkpoint['mode'],
                config=checkpoint['config']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.metrics = checkpoint['metrics']
            return model

        def add_expert(
            self,
            expert_id: str,
            expert: Expert,
            modality: Optional[str] = None
        ):
            """Add a new expert to the model."""
            if self.mode == "modality":
                self.model.add_expert(expert_id, expert, modality)
            else:
                raise ValueError("Adding experts only available for modality MoE")

        def remove_expert(self, expert_id: str):
            """Remove an expert from the model."""
            if self.mode == "modality":
                self.model.remove_expert(expert_id)
            else:
                raise ValueError("Removing experts only available for modality MoE")

        def get_expert_info(self) -> Dict[str, Any]:
            """Get information about all experts."""
            if self.mode == "modality":
                return self.model.get_expert_info()
            else:
                return {"mode": "neural", "num_experts": self.config.get('num_experts', 8)}

else:
    class UnifiedMoE:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for UnifiedMoE. Please install torch.") 