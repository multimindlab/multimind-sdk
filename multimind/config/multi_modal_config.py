"""
Configuration management for multi-modal models and settings.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import yaml

class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    name: str
    type: str
    modality: str
    cost_per_token: float = 0.0
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)

class MoEConfig(BaseModel):
    """Configuration for MoE model."""
    hidden_size: int = 768
    num_experts: int = 4
    expert_threshold: float = 0.1
    fusion_type: str = "concatenate"

class RouterConfig(BaseModel):
    """Configuration for model router."""
    cost_weight: float = 0.7
    performance_weight: float = 0.3
    switch_threshold: float = 0.8
    max_retries: int = 3

class MultiModalConfig(BaseModel):
    """Main configuration for multi-modal processing."""
    models: Dict[str, ModelConfig]
    moe: MoEConfig = Field(default_factory=MoEConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    default_workflow: Optional[str] = None

class ConfigManager:
    """Manager for multi-modal configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config: Optional[MultiModalConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file or use defaults."""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                self.config = MultiModalConfig(**config_data)
        else:
            # Use default configuration
            self.config = MultiModalConfig(
                models={},
                moe=MoEConfig(),
                router=RouterConfig()
            )
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        if not self.config:
            return
        
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No path provided for saving configuration")
        
        config_data = self.config.dict()
        with open(save_path, 'w') as f:
            yaml.dump(config_data, f)
    
    def add_model(self, model_config: ModelConfig) -> None:
        """Add a model configuration."""
        if not self.config:
            self.config = MultiModalConfig(models={})
        self.config.models[model_config.name] = model_config
    
    def remove_model(self, model_name: str) -> None:
        """Remove a model configuration."""
        if self.config and model_name in self.config.models:
            del self.config.models[model_name]
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        if self.config and model_name in self.config.models:
            return self.config.models[model_name]
        return None
    
    def update_moe_config(self, moe_config: MoEConfig) -> None:
        """Update MoE configuration."""
        if self.config:
            self.config.moe = moe_config
    
    def update_router_config(self, router_config: RouterConfig) -> None:
        """Update router configuration."""
        if self.config:
            self.config.router = router_config
    
    def get_config(self) -> MultiModalConfig:
        """Get the current configuration."""
        if not self.config:
            self._load_config()
        return self.config

# Create global config manager instance
config_manager = ConfigManager()

def get_config() -> ConfigManager:
    """Get the global configuration manager."""
    return config_manager 