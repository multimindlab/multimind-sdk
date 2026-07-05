"""
Factory for creating MoE (Mixture of Experts) models.
"""

import logging
from typing import Any, Dict, Optional

from .moe_model import MoEModel

logger = logging.getLogger(__name__)


class MoEFactory:
    """Factory for creating MoE model instances."""

    def __init__(self):
        """Initialize the MoE factory."""
        self._models: Dict[str, MoEModel] = {}
        logger.info("MoE Factory initialized")

    def create_moe_model(self, config: Dict[str, Any]) -> MoEModel:
        """
        Create a new MoE model instance.

        Args:
            config: Configuration dictionary for the MoE model

        Returns:
            MoEModel instance
        """
        try:
            model = MoEModel(config)
            model_id = id(model)
            self._models[str(model_id)] = model
            logger.info(f"Created MoE model with ID: {model_id}")
            return model
        except Exception as e:
            logger.error(f"Failed to create MoE model: {e}")
            raise

    def get_model(self, model_id: str) -> Optional[MoEModel]:
        """
        Get an existing MoE model by ID.

        Args:
            model_id: Model identifier

        Returns:
            MoEModel instance or None if not found
        """
        return self._models.get(model_id)

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all created MoE models.

        Returns:
            Dictionary of model information
        """
        result: Dict[str, Dict[str, Any]] = {}

        for model_id, model in self._models.items():
            # config: prefer `model.config`, fallback to `model.get_config()`
            if hasattr(model, "config"):
                config = model.config
            elif hasattr(model, "get_config") and callable(model.get_config):
                config = model.get_config()
            else:
                config = None

            # experts: if an `experts` dict exists, return its keys; otherwise derive from `num_experts` when possible.
            experts: list = []
            if hasattr(model, "experts"):
                exp = model.experts
                if isinstance(exp, dict):
                    experts = list(exp.keys())
                elif isinstance(exp, (list, tuple, set)):
                    experts = list(exp)

            if not experts and hasattr(model, "num_experts"):
                try:
                    n = int(model.num_experts)
                    experts = [f"expert_{i}" for i in range(n)]
                except Exception:
                    experts = []

            # gateway name if present
            gateway = None
            if hasattr(model, "gateway"):
                gw = model.gateway
                gateway = gw.__class__.__name__ if gw is not None else None

            result[model_id] = {
                "config": config,
                "experts": experts,
                "gateway": gateway,
            }

        return result

    def remove_model(self, model_id: str) -> bool:
        """
        Remove a MoE model.

        Args:
            model_id: Model identifier

        Returns:
            True if removed, False if not found
        """
        if model_id in self._models:
            del self._models[model_id]
            logger.info(f"Removed MoE model: {model_id}")
            return True
        return False
