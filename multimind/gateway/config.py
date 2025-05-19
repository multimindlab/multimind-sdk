"""
Configuration module for the MultiMind Gateway
"""

import os
from typing import Dict, Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ModelConfig(BaseSettings):
    """Configuration for individual models"""
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: int = 30

class GatewayConfig(BaseSettings):
    """Main configuration for the MultiMind Gateway"""
    
    # OpenAI Configuration
    openai: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        )
    )
    
    # Anthropic Configuration
    anthropic: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model_name=os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-opus-20240229")
        )
    )
    
    # Ollama Configuration
    ollama: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
            model_name=os.getenv("OLLAMA_MODEL_NAME", "mistral")
        )
    )
    
    # Groq Configuration
    groq: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL_NAME", "mixtral-8x7b-32768")
        )
    )
    
    # HuggingFace Configuration
    huggingface: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("HUGGINGFACE_API_KEY"),
            model_name=os.getenv("HUGGINGFACE_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
        )
    )
    
    # General Settings
    default_model: str = Field(
        default=os.getenv("DEFAULT_MODEL", "openai"),
        description="Default model to use when none specified"
    )
    
    log_level: str = Field(
        default=os.getenv("LOG_LEVEL", "INFO"),
        description="Logging level for the gateway"
    )
    
    class Config:
        env_prefix = "MULTIMIND_"
        case_sensitive = False

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model"""
        model_map = {
            "openai": self.openai,
            "anthropic": self.anthropic,
            "ollama": self.ollama,
            "groq": self.groq,
            "huggingface": self.huggingface
        }
        return model_map.get(model_name.lower(), self.openai)

    def validate(self) -> Dict[str, bool]:
        """Validate the configuration and return status for each model"""
        validation_status = {}
        
        # Check OpenAI
        validation_status["openai"] = bool(self.openai.api_key)
        
        # Check Anthropic
        validation_status["anthropic"] = bool(self.anthropic.api_key)
        
        # Check Ollama (only needs api_base)
        validation_status["ollama"] = bool(self.ollama.api_base)
        
        # Check Groq
        validation_status["groq"] = bool(self.groq.api_key)
        
        # Check HuggingFace
        validation_status["huggingface"] = bool(self.huggingface.api_key)
        
        return validation_status

# Create a global config instance
config = GatewayConfig() 