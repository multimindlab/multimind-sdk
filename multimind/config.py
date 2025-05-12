"""
Configuration settings for the Multimind SDK.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Configuration for individual model providers."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SDKConfig:
    """Global SDK configuration."""
    default_model: str = "gpt-3.5-turbo"
    cache_dir: str = os.path.expanduser("~/.multimind/cache")
    log_level: str = "INFO"
    enable_logging: bool = True
    models: Dict[str, ModelConfig] = field(default_factory=dict)

    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)

# Global configuration instance
config = SDKConfig() 