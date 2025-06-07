import os
import json
import requests
from typing import Dict, Any, Optional
from .base import BaseModelConverter

class OllamaConverter(BaseModelConverter):
    """Converter for Ollama models."""
    
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.supported_formats = ["gguf"]
    
    def convert(self, 
                model_path: str,
                output_path: str,
                config: Optional[Dict[str, Any]] = None) -> str:
        """
        Convert a model to Ollama format.
        
        Args:
            model_path: Path to the source model
            output_path: Path where the converted model should be saved
            config: Optional configuration parameters for the conversion
            
        Returns:
            str: Path to the converted model
        """
        if not self.validate(model_path):
            raise ValueError(f"Invalid model path: {model_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Prepare model configuration
        model_config = {
            "name": os.path.basename(output_path),
            "path": model_path,
            **(config or {})
        }
        
        # Create Modelfile
        modelfile_path = os.path.join(output_path, "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(f"FROM {model_path}\n")
            if config:
                for key, value in config.items():
                    f.write(f"PARAMETER {key} {value}\n")
        
        # Create model using Ollama API
        response = requests.post(
            f"{self.ollama_host}/api/create",
            json=model_config
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to create Ollama model: {response.text}")
        
        return output_path
    
    def validate(self, model_path: str) -> bool:
        """
        Validate if the model can be converted.
        
        Args:
            model_path: Path to the model to validate
            
        Returns:
            bool: True if the model can be converted, False otherwise
        """
        try:
            # Check if Ollama server is running
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code != 200:
                return False
            
            # Check if model file exists
            return os.path.exists(model_path)
        except Exception:
            return False
    
    def get_metadata(self, model_path: str) -> Dict[str, Any]:
        """
        Get metadata about the model.
        
        Args:
            model_path: Path to the model
            
        Returns:
            Dict[str, Any]: Model metadata
        """
        try:
            response = requests.get(
                f"{self.ollama_host}/api/show",
                json={"name": os.path.basename(model_path)}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": "Failed to get model metadata",
                    "status_code": response.status_code
                }
        except Exception as e:
            return {
                "error": str(e),
                "status_code": 500
            } 