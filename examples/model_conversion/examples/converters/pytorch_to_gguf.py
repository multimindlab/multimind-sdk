#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import Dict, Any
import torch
from multimind.model_conversion import ModelConversionManager

def convert_pytorch_to_gguf(
    model_path: str,
    output_path: str,
    config: Dict[str, Any]
) -> str:
    """Convert PyTorch model to GGUF format with advanced options."""
    converter = ModelConversionManager(
        source_format="pytorch",
        target_format="gguf",
        pipeline_config={
            "pipeline": [
                {
                    "step": "validate",
                    "checks": ["format", "metadata"]
                },
                {
                    "step": "optimize",
                    "methods": ["pruning", "quantization"],
                    "config": {
                        "pruning": {"sparsity": config.get("sparsity", 0.3)},
                        "quantization": {
                            "method": config.get("quantization", "q4_k_m"),
                            "calibration_data": config.get("calibration_data")
                        }
                    }
                },
                {
                    "step": "convert",
                    "format_options": {
                        "gguf": {
                            "context_length": config.get("context_length", 4096),
                            "embedding_type": config.get("embedding_type", "float32"),
                            "use_mlock": config.get("use_mlock", True)
                        }
                    }
                }
            ]
        }
    )
    
    return converter.convert(model_path, output_path)

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to GGUF format")
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to PyTorch model")
    parser.add_argument("--output-path", type=str, required=True,
                      help="Path to save GGUF model")
    parser.add_argument("--sparsity", type=float, default=0.3,
                      help="Pruning sparsity (0.0 to 1.0)")
    parser.add_argument("--quantization", type=str, default="q4_k_m",
                      choices=["q4_k_m", "q4_0", "q5_k_m", "q8_0"],
                      help="Quantization method")
    parser.add_argument("--context-length", type=int, default=4096,
                      help="Model context length")
    parser.add_argument("--embedding-type", type=str, default="float32",
                      choices=["float32", "float16", "int8"],
                      help="Embedding type")
    
    args = parser.parse_args()
    
    config = {
        "sparsity": args.sparsity,
        "quantization": args.quantization,
        "context_length": args.context_length,
        "embedding_type": args.embedding_type
    }
    
    try:
        output_path = convert_pytorch_to_gguf(
            args.model_path,
            args.output_path,
            config
        )
        print(f"Model converted successfully to: {output_path}")
        
        # Print model metadata
        converter = ModelConversionManager()
        metadata = converter.get_metadata(output_path)
        print("\nModel Metadata:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 