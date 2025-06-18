"""
Example demonstrating model registration and configuration for the multi-modal API.
"""

import asyncio
import os
from typing import Dict, Any
from multimind.config.multi_modal_config import (
    ModelConfig,
    MoEConfig,
    RouterConfig,
    get_config
)
from multimind.router.multi_modal_router import MultiModalRouter
from multimind.models.moe import MoEFactory

async def register_models():
    """Register models for different modalities."""
    # Get config manager
    config_manager = get_config()
    
    # Configure text models
    text_models = {
        "gpt4": ModelConfig(
            name="gpt4",
            type="openai",
            modality="text",
            cost_per_token=0.03,
            max_tokens=8192,
            temperature=0.7,
            api_key="your-api-key"  # Replace with actual key
        ),
        "claude": ModelConfig(
            name="claude",
            type="anthropic",
            modality="text",
            cost_per_token=0.02,
            max_tokens=100000,
            temperature=0.7,
            api_key="your-api-key"  # Replace with actual key
        )
    }
    
    # Configure vision models
    vision_models = {
        "gpt4v": ModelConfig(
            name="gpt4v",
            type="openai",
            modality="vision",
            cost_per_token=0.03,
            max_tokens=4096,
            temperature=0.7,
            api_key="your-api-key"  # Replace with actual key
        ),
        "clip": ModelConfig(
            name="clip",
            type="huggingface",
            modality="vision",
            cost_per_token=0.0,  # Local model
            max_tokens=None,
            temperature=0.7
        )
    }
    
    # Configure audio models
    audio_models = {
        "whisper": ModelConfig(
            name="whisper",
            type="openai",
            modality="audio",
            cost_per_token=0.006,
            max_tokens=None,
            temperature=0.7,
            api_key="your-api-key"  # Replace with actual key
        )
    }
    
    # Register all models
    for model in [*text_models.values(), *vision_models.values(), *audio_models.values()]:
        config_manager.add_model(model)
    
    # Configure MoE settings
    moe_config = MoEConfig(
        hidden_size=1024,
        num_experts=5,
        expert_threshold=0.1,
        fusion_type="concatenate"
    )
    config_manager.update_moe_config(moe_config)
    
    # Configure router settings
    router_config = RouterConfig(
        cost_weight=0.7,
        performance_weight=0.3,
        switch_threshold=0.8,
        max_retries=3
    )
    config_manager.update_router_config(router_config)
    
    # Create config directory if it doesn't exist
    config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "config")
    os.makedirs(config_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config_dir, "multi_modal_config.yaml")
    config_manager.save_config(config_path)
    
    print("Models registered successfully!")
    print(f"Total models: {len(config_manager.get_config().models)}")
    print("\nRegistered models by modality:")
    for model in config_manager.get_config().models.values():
        print(f"- {model.name} ({model.modality})")

if __name__ == "__main__":
    asyncio.run(register_models()) 