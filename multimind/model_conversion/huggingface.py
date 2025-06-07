import os
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseModelConverter

class HuggingFaceConverter(BaseModelConverter):
    """Converter for HuggingFace models."""
    
    def __init__(self):
        self.supported_formats = ["pytorch", "safetensors"]
    
    def convert(self, 
                model_path: str,
                output_path: str,
                config: Optional[Dict[str, Any]] = None) -> str:
        """
        Convert a HuggingFace model to the specified format.
        
        Args:
            model_path: Path to the HuggingFace model
            output_path: Path where the converted model should be saved
            config: Optional configuration parameters for the conversion
            
        Returns:
            str: Path to the converted model
        """
        if not self.validate(model_path):
            raise ValueError(f"Invalid model path: {model_path}")
            
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
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
            # Try to load the model and tokenizer
            AutoModelForCausalLM.from_pretrained(model_path)
            AutoTokenizer.from_pretrained(model_path)
            return True
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
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return {
            "model_type": model.config.model_type,
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "num_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "tokenizer_type": tokenizer.__class__.__name__,
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        } 