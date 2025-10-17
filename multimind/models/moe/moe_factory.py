"""
Factory for creating MoE (Mixture of Experts) models.
"""

import logging
from typing import Dict, Any, Optional
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
        return {
            model_id: {
                "config": model.config,
                "experts": list(model.experts.keys()),
                "gateway": model.gateway.__class__.__name__ if model.gateway else None
            }
            for model_id, model in self._models.items()
        }
    
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
