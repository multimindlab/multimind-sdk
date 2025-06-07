#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import Dict, Any
import torch
from safetensors.torch import save_file
from multimind.model_conversion import ModelConversionManager

def convert_pytorch_to_safetensors(
    model_path: str,
    output_path: str,
    config: Dict[str, Any]
) -> str:
    """Convert PyTorch model to Safetensors format with advanced options."""
    converter = ModelConversionManager(
        source_format="pytorch",
        target_format="safetensors",
        pipeline_config={
            "pipeline": [
                {
                    "step": "validate",
                    "checks": ["format", "metadata"]
                },
                {
                    "step": "optimize",
                    "methods": ["compression"],
                    "config": {
                        "compression": {
                            "method": config.get("compression", "lz4"),
                            "compression_level": config.get("compression_level", 9)
                        }
                    }
                },
                {
                    "step": "convert",
                    "format_options": {
                        "safetensors": {
                            "metadata": config.get("metadata", {}),
                            "force_contiguous": config.get("force_contiguous", True),
                            "device": config.get("device", "cpu")
                        }
                    }
                }
            ]
        }
    )
    
    return converter.convert(model_path, output_path)

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to Safetensors format")
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to PyTorch model")
    parser.add_argument("--output-path", type=str, required=True,
                      help="Path to save Safetensors model")
    parser.add_argument("--compression", type=str, default="lz4",
                      choices=["lz4", "zstd", "none"],
                      help="Compression method")
    parser.add_argument("--compression-level", type=int, default=9,
                      help="Compression level (1-9)")
    parser.add_argument("--device", type=str, default="cpu",
                      choices=["cpu", "cuda"],
                      help="Device to use for conversion")
    parser.add_argument("--metadata", type=str, nargs="+",
                      help="Additional metadata as key=value pairs")
    
    args = parser.parse_args()
    
    # Parse metadata
    metadata = {}
    if args.metadata:
        for item in args.metadata:
            key, value = item.split("=")
            metadata[key] = value
    
    config = {
        "compression": args.compression,
        "compression_level": args.compression_level,
        "device": args.device,
        "metadata": metadata
    }
    
    try:
        output_path = convert_pytorch_to_safetensors(
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