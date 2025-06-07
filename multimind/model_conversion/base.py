from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

class BaseModelConverter(ABC):
    """Base class for model converters."""
    
    @abstractmethod
    def convert(self, 
                model_path: str,
                output_path: str,
                config: Optional[Dict[str, Any]] = None) -> str:
        """
        Convert a model from source format to target format.
        
        Args:
            model_path: Path to the source model
            output_path: Path where the converted model should be saved
            config: Optional configuration parameters for the conversion
            
        Returns:
            str: Path to the converted model
        """
        pass
    
    @abstractmethod
    def validate(self, model_path: str) -> bool:
        """
        Validate if the model can be converted.
        
        Args:
            model_path: Path to the model to validate
            
        Returns:
            bool: True if the model can be converted, False otherwise
        """
        pass
    
    @abstractmethod
    def get_metadata(self, model_path: str) -> Dict[str, Any]:
        """
        Get metadata about the model.
        
        Args:
            model_path: Path to the model
            
        Returns:
            Dict[str, Any]: Model metadata
        """
        pass 