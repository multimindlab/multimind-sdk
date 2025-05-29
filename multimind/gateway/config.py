"""
Configuration module for the MultiMind Gateway API
"""

from ..core.config import GatewayConfig, ModelConfig, config

# Re-export the config instance for API use
__all__ = ['config', 'GatewayConfig', 'ModelConfig']