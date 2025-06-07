#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import Dict, Any
import tensorflow as tf
import onnx
from multimind.model_conversion import ModelConversionManager

def convert_tensorflow_to_onnx(
    model_path: str,
    output_path: str,
    config: Dict[str, Any]
) -> str:
    """Convert TensorFlow model to ONNX format with advanced options."""
    converter = ModelConversionManager(
        source_format="tensorflow",
        target_format="onnx",
        pipeline_config={
            "pipeline": [
                {
                    "step": "validate",
                    "checks": ["format", "metadata", "compatibility"]
                },
                {
                    "step": "optimize",
                    "methods": ["graph_optimization"],
                    "config": {
                        "graph_optimization": {
                            "opset_version": config.get("opset_version", 12),
                            "custom_ops": config.get("custom_ops", {}),
                            "input_signature": config.get("input_signature"),
                            "output_signature": config.get("output_signature")
                        }
                    }
                },
                {
                    "step": "convert",
                    "format_options": {
                        "onnx": {
                            "dynamic_axes": config.get("dynamic_axes", {}),
                            "do_constant_folding": config.get("do_constant_folding", True),
                            "verbose": config.get("verbose", False)
                        }
                    }
                }
            ]
        }
    )
    
    return converter.convert(model_path, output_path)

def main():
    parser = argparse.ArgumentParser(description="Convert TensorFlow model to ONNX format")
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to TensorFlow model")
    parser.add_argument("--output-path", type=str, required=True,
                      help="Path to save ONNX model")
    parser.add_argument("--opset-version", type=int, default=12,
                      help="ONNX opset version")
    parser.add_argument("--do-constant-folding", action="store_true",
                      help="Enable constant folding")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output")
    parser.add_argument("--input-shape", type=str, nargs="+",
                      help="Input shapes as name=shape pairs")
    parser.add_argument("--output-shape", type=str, nargs="+",
                      help="Output shapes as name=shape pairs")
    
    args = parser.parse_args()
    
    # Parse input and output shapes
    input_signature = {}
    if args.input_shape:
        for item in args.input_shape:
            name, shape = item.split("=")
            input_signature[name] = [int(dim) for dim in shape.split(",")]
    
    output_signature = {}
    if args.output_shape:
        for item in args.output_shape:
            name, shape = item.split("=")
            output_signature[name] = [int(dim) for dim in shape.split(",")]
    
    config = {
        "opset_version": args.opset_version,
        "do_constant_folding": args.do_constant_folding,
        "verbose": args.verbose,
        "input_signature": input_signature,
        "output_signature": output_signature
    }
    
    try:
        output_path = convert_tensorflow_to_onnx(
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