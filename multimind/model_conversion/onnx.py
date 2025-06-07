import os
import onnx
import torch
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseModelConverter

class ONNXConverter(BaseModelConverter):
    """Converter for ONNX models."""
    
    def __init__(self):
        self.supported_formats = ["onnx"]
        self.required_dependencies = ["onnx", "onnxruntime"]
    
    def convert(self, 
                model_path: str,
                output_path: str,
                config: Optional[Dict[str, Any]] = None) -> str:
        """
        Convert a model to ONNX format.
        
        Args:
            model_path: Path to the source model
            output_path: Path where the converted model should be saved
            config: Optional configuration parameters for the conversion
                   Supported config options:
                   - opset_version: ONNX opset version (default: 12)
                   - dynamic_axes: Dynamic axes configuration
                   - input_names: Input tensor names
                   - output_names: Output tensor names
                   - device: Device to use for conversion (default: "cpu")
        
        Returns:
            str: Path to the converted model
        """
        if not self.validate(model_path):
            raise ValueError(f"Invalid model path: {model_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set default config values
        config = config or {}
        opset_version = config.get("opset_version", 12)
        device = config.get("device", "cpu")
        
        # Prepare dynamic axes configuration
        dynamic_axes = config.get("dynamic_axes", {
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "output": {0: "batch_size", 1: "sequence"}
        })
        
        # Prepare input names
        input_names = config.get("input_names", ["input_ids", "attention_mask"])
        output_names = config.get("output_names", ["output"])
        
        # Create dummy input for tracing
        dummy_input = {
            "input_ids": torch.ones(1, 10, dtype=torch.long, device=device),
            "attention_mask": torch.ones(1, 10, dtype=torch.long, device=device)
        }
        
        # Export model to ONNX
        onnx_path = os.path.join(output_path, "model.onnx")
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(output_path)
        
        # Save model configuration
        model.config.save_pretrained(output_path)
        
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
            # Check if required dependencies are installed
            import onnx
            import onnxruntime
            
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
        
        # Get ONNX-specific metadata if the model is already in ONNX format
        onnx_metadata = {}
        onnx_path = os.path.join(model_path, "model.onnx")
        if os.path.exists(onnx_path):
            onnx_model = onnx.load(onnx_path)
            onnx_metadata = {
                "onnx_version": onnx_model.ir_version,
                "producer_name": onnx_model.producer_name,
                "producer_version": onnx_model.producer_version,
                "opset_version": onnx_model.opset_import[0].version,
                "input_shapes": [input.type.tensor_type.shape for input in onnx_model.graph.input],
                "output_shapes": [output.type.tensor_type.shape for output in onnx_model.graph.output]
            }
        
        return {
            "model_type": model.config.model_type,
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "num_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "tokenizer_type": tokenizer.__class__.__name__,
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
            "onnx_metadata": onnx_metadata
        } 