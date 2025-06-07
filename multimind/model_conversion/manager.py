import os
from typing import Dict, Any, Optional, List, Type
from .base import BaseModelConverter
from .huggingface import HuggingFaceConverter
from .ollama import OllamaConverter

class ModelConversionManager:
    """Manager for model conversion operations."""
    
    def __init__(self):
        self.converters: Dict[str, BaseModelConverter] = {
            "huggingface": HuggingFaceConverter(),
            "ollama": OllamaConverter()
        }
    
    def register_converter(self, name: str, converter: BaseModelConverter) -> None:
        """
        Register a new converter.
        
        Args:
            name: Name of the converter
            converter: Converter instance
        """
        self.converters[name] = converter
    
    def get_converter(self, name: str) -> BaseModelConverter:
        """
        Get a converter by name.
        
        Args:
            name: Name of the converter
            
        Returns:
            BaseModelConverter: The converter instance
            
        Raises:
            KeyError: If the converter is not found
        """
        if name not in self.converters:
            raise KeyError(f"Converter '{name}' not found")
        return self.converters[name]
    
    def list_converters(self) -> List[str]:
        """
        List all available converters.
        
        Returns:
            List[str]: List of converter names
        """
        return list(self.converters.keys())
    
    def convert(self,
                model_path: str,
                output_path: str,
                converter_name: str,
                config: Optional[Dict[str, Any]] = None) -> str:
        """
        Convert a model using the specified converter.
        
        Args:
            model_path: Path to the source model
            output_path: Path where the converted model should be saved
            converter_name: Name of the converter to use
            config: Optional configuration parameters for the conversion
            
        Returns:
            str: Path to the converted model
            
        Raises:
            KeyError: If the converter is not found
            ValueError: If the model path is invalid
        """
        converter = self.get_converter(converter_name)
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        return converter.convert(model_path, output_path, config)
    
    def validate_model(self, model_path: str, converter_name: str) -> bool:
        """
        Validate if a model can be converted using the specified converter.
        
        Args:
            model_path: Path to the model to validate
            converter_name: Name of the converter to use
            
        Returns:
            bool: True if the model can be converted, False otherwise
        """
        converter = self.get_converter(converter_name)
        return converter.validate(model_path)
    
    def get_model_metadata(self, model_path: str, converter_name: str) -> Dict[str, Any]:
        """
        Get metadata about a model using the specified converter.
        
        Args:
            model_path: Path to the model
            converter_name: Name of the converter to use
            
        Returns:
            Dict[str, Any]: Model metadata
        """
        converter = self.get_converter(converter_name)
        return converter.get_metadata(model_path) 