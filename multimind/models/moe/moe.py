"""
Base classes for Mixture of Experts (MoE) implementation.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Optional torch import for MoE base features
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. MoE base features will be disabled.")

import numpy as np


class Expert(ABC):
    """Abstract base class for experts in MoE."""

    def __init__(self, expert_id: str, **kwargs):
        self.expert_id = expert_id
        self.kwargs = kwargs
        self.usage_count = 0
        self.performance_metrics = {}

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input data and return output."""
        pass

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics."""
        self.performance_metrics.update(metrics)
        self.usage_count += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {"usage_count": self.usage_count, "performance_metrics": self.performance_metrics}


class ExpertRouter(ABC):
    """Abstract base class for expert routing."""

    def __init__(self, experts: Dict[str, Expert], **kwargs):
        self.experts = experts
        self.kwargs = kwargs
        self.routing_history = []
        # Prevent unbounded growth in high-throughput systems.
        self.max_routing_history: int = int(kwargs.get("max_routing_history", 1000))

    @abstractmethod
    async def route(self, input_data: Any) -> Dict[str, float]:
        """Route input to experts and return weights."""
        pass

    def update_routing_history(self, input_data: Any, weights: Dict[str, float]):
        """Update routing history."""
        # Use a real wall-clock timestamp; CUDA timing events are not general timestamps.
        self.routing_history.append(
            {
                "input": input_data,
                "weights": weights,
                "timestamp": datetime.now().isoformat(),
            }
        )
        if self.max_routing_history > 0 and len(self.routing_history) > self.max_routing_history:
            # Keep only the most recent entries.
            self.routing_history = self.routing_history[-self.max_routing_history :]

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if not self.routing_history:
            return {}

        # Calculate average weights for each expert
        avg_weights = {}
        for expert_id in self.experts.keys():
            weights = [entry["weights"].get(expert_id, 0.0) for entry in self.routing_history]
            avg_weights[expert_id] = np.mean(weights)

        return {"total_routes": len(self.routing_history), "average_weights": avg_weights}


if TORCH_AVAILABLE:

    class MoEBase(nn.Module):
        """Base class for Mixture of Experts models."""

        def __init__(
            self,
            experts: Dict[str, Expert],
            hidden_size: int = 768,
            num_experts: Optional[int] = None,
            **kwargs,
        ):
            super().__init__()
            self.experts = experts
            self.hidden_size = hidden_size
            self.num_experts = num_experts or len(experts)
            self.kwargs = kwargs

            # Initialize a concrete router implementation (ExpertRouter is abstract).
            self.router = ModalityRouter(experts, **kwargs)

            # Initialize metrics
            self.metrics = {
                "expert_usage": {expert_id: 0 for expert_id in experts},
                "routing_weights": {},
                "performance_metrics": {},
            }

        async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process input through the MoE model."""
            # Route input to experts
            weights = await self.router.route(input_data)

            # Update routing history
            self.router.update_routing_history(input_data, weights)

            # Process with each expert
            expert_outputs = {}
            for expert_id, weight in weights.items():
                if weight > 0.0 and expert_id in self.experts:
                    expert = self.experts[expert_id]
                    # Pass only the matching modality payload to each expert when possible.
                    expert_input: Any = input_data
                    if isinstance(input_data, dict):
                        expert_type = expert.__class__.__name__.lower()
                        expert_key = expert_id.lower()
                        for modality in input_data:
                            m = str(modality).lower()
                            if m in expert_type or m in expert_key:
                                # For non-text experts, include text prompt if available.
                                if m in ("image", "audio") and "text" in input_data:
                                    expert_input = {
                                        "text": input_data.get("text"),
                                        modality: input_data.get(modality),
                                    }
                                else:
                                    expert_input = input_data.get(modality)
                                break
                    output = await expert.process(expert_input)
                    expert_outputs[expert_id] = {"output": output, "weight": weight}

                    # Update metrics
                    self.metrics["expert_usage"][expert_id] += 1

            # Combine expert outputs
            combined_output = self._combine_outputs(expert_outputs)

            # Update metrics
            self.metrics["routing_weights"] = weights
            self.metrics["performance_metrics"] = {
                expert_id: expert.get_metrics() for expert_id, expert in self.experts.items()
            }

            return {
                "output": combined_output,
                "expert_outputs": expert_outputs,
                "routing_weights": weights,
                "metrics": self.metrics,
            }

        def _combine_outputs(self, expert_outputs: Dict[str, Dict[str, Any]]) -> Any:
            """Combine outputs from multiple experts."""
            if not expert_outputs:
                return "No expert produced output."

            total_weight = sum(output["weight"] for output in expert_outputs.values())
            normalized = []
            for expert_data in expert_outputs.values():
                weight = expert_data["weight"] / total_weight if total_weight > 0 else 0.0
                normalized.append((weight, expert_data["output"]))

            outputs = [o for _, o in normalized]
            if all(isinstance(o, str) for o in outputs):
                parts = [str(o).strip() for _, o in normalized if str(o).strip()]
                return "\n\n".join(parts) if parts else ""
            if all(torch.is_tensor(o) for o in outputs):
                combined = torch.zeros_like(outputs[0])
                for weight, output in normalized:
                    combined += weight * output
                return combined

            if all(isinstance(o, (int, float)) for o in outputs):
                return sum(weight * output for weight, output in normalized)

            # For structured outputs, return highest-weight expert output.
            return max(normalized, key=lambda item: item[0])[1]

        def get_metrics(self) -> Dict[str, Any]:
            """Get current metrics."""
            return self.metrics

        def reset_metrics(self):
            """Reset all metrics."""
            self.metrics = {
                "expert_usage": {expert_id: 0 for expert_id in self.experts.keys()},
                "routing_weights": {},
                "performance_metrics": {},
            }
            for expert in self.experts.values():
                expert.usage_count = 0
                expert.performance_metrics = {}

else:

    class MoEBase:
        """Base class for Mixture of Experts models."""

        def __init__(
            self,
            experts: Dict[str, Expert],
            hidden_size: int = 768,
            num_experts: Optional[int] = None,
            **kwargs,
        ):
            raise ImportError("PyTorch is required for MoEBase. Please install torch.")


class TextExpert(Expert):
    """Text processing expert."""

    def __init__(self, expert_id: str, model_name: str = "gpt2", **kwargs):
        super().__init__(expert_id, **kwargs)
        self.model_name = model_name

    async def process(self, input_data: str) -> str:
        """Process text input."""
        # Placeholder implementation
        return f"Processed text with {self.model_name}: {input_data}"


class ImageExpert(Expert):
    """Image processing expert."""

    def __init__(self, expert_id: str, model_name: str = "resnet", **kwargs):
        super().__init__(expert_id, **kwargs)
        self.model_name = model_name

    async def process(self, input_data: Any) -> Any:
        """Process image input."""
        # Placeholder implementation
        return f"Processed image with {self.model_name}"


class AudioExpert(Expert):
    """Audio processing expert."""

    def __init__(self, expert_id: str, model_name: str = "wav2vec", **kwargs):
        super().__init__(expert_id, **kwargs)
        self.model_name = model_name

    async def process(self, input_data: Any) -> Any:
        """Process audio input."""
        # Placeholder implementation
        return f"Processed audio with {self.model_name}"


class SimpleRouter(ExpertRouter):
    """Simple expert router that distributes load evenly."""

    async def route(self, input_data: Any) -> Dict[str, float]:
        """Route input to all experts with equal weights."""
        num_experts = len(self.experts)
        weight = 1.0 / num_experts if num_experts > 0 else 0.0
        return {expert_id: weight for expert_id in self.experts.keys()}


class ModalityRouter(ExpertRouter):
    """Router that routes based on input modality."""

    async def route(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Rule-based routing with equal weights among matched experts.

        Behavior:
        - If 3 experts match -> 0.33 each
        - If 2 experts match -> 0.50 each
        - If 1 expert matches -> 1.00
        """
        detected_modalities = [str(k).lower() for k in input_data]

        weights: Dict[str, float] = {}
        for expert_id, expert in self.experts.items():
            expert_type = expert.__class__.__name__.lower()
            expert_key = str(expert_id).lower()

            match = any(
                (modality in expert_type) or (modality in expert_key)
                for modality in detected_modalities
            )
            weights[expert_id] = 1.0 if match else 0.0

        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}

        # If nothing matches, return all zeros (explicitly "no route").
        return weights

    def _detect_modality(self, input_data: Dict[str, Any]) -> str:
        """Detect the modality of input data."""
        if "text" in input_data:
            return "text"
        elif "image" in input_data:
            return "image"
        elif "audio" in input_data:
            return "audio"
        else:
            return "unknown"

    def _expert_matches_modality(self, expert: Expert, modality: str) -> bool:
        """Check if expert matches the detected modality."""
        expert_type = expert.__class__.__name__.lower()
        if (
            modality == "text"
            and "text" in expert_type
            or modality == "image"
            and "image" in expert_type
            or modality == "audio"
            and "audio" in expert_type
        ):
            return True
        return False
