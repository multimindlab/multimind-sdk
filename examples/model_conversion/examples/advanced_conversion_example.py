#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List
import torch
import tensorflow as tf
from multimind.model_conversion import ModelConversionManager
from multimind.model_conversion.formats import (
    TensorFlowConverter,
    ONNXRuntimeConverter,
    SafetensorsConverter,
    GGMLConverter
)

def setup_conversion_pipeline() -> Dict[str, Any]:
    """Set up the conversion pipeline configuration."""
    return {
        "pipeline": [
            {
                "step": "validate",
                "checks": ["format", "metadata", "compatibility"]
            },
            {
                "step": "optimize",
                "methods": ["pruning", "quantization"],
                "config": {
                    "pruning": {"sparsity": 0.3},
                    "quantization": {"method": "dynamic"}
                }
            },
            {
                "step": "convert",
                "target_formats": ["gguf", "onnx", "tflite"],
                "parallel": True
            }
        ]
    }

def convert_pytorch_to_multiple_formats(
    model_path: str,
    output_dir: str,
    config: Dict[str, Any]
) -> List[str]:
    """Convert PyTorch model to multiple formats."""
    converter = ModelConversionManager(
        source_format="pytorch",
        target_formats=["gguf", "onnx", "safetensors"],
        pipeline_config=config
    )
    
    return converter.convert(model_path, output_dir)

def convert_tensorflow_to_multiple_formats(
    model_path: str,
    output_dir: str,
    config: Dict[str, Any]
) -> List[str]:
    """Convert TensorFlow model to multiple formats."""
    converter = ModelConversionManager(
        source_format="tensorflow",
        target_formats=["tflite", "onnx"],
        pipeline_config=config
    )
    
    return converter.convert(model_path, output_dir)

def convert_onnx_to_optimized_runtime(
    model_path: str,
    output_dir: str,
    config: Dict[str, Any]
) -> str:
    """Convert ONNX model to optimized runtime format."""
    converter = ONNXRuntimeConverter()
    return converter.convert(model_path, output_dir, config)

def convert_to_safetensors(
    model_path: str,
    output_dir: str,
    source_format: str,
    config: Dict[str, Any]
) -> str:
    """Convert model to Safetensors format."""
    converter = SafetensorsConverter()
    config["source_format"] = source_format
    return converter.convert(model_path, output_dir, config)

def main():
    parser = argparse.ArgumentParser(description="Advanced model conversion example")
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to the source model")
    parser.add_argument("--output-dir", type=str, default="./converted_models",
                      help="Directory to save converted models")
    parser.add_argument("--source-format", type=str, required=True,
                      choices=["pytorch", "tensorflow", "onnx"],
                      help="Source model format")
    parser.add_argument("--target-formats", type=str, nargs="+",
                      default=["gguf", "onnx", "tflite", "safetensors"],
                      help="Target formats for conversion")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up conversion pipeline
    pipeline_config = setup_conversion_pipeline()
    
    try:
        # Perform conversions based on source format
        if args.source_format == "pytorch":
            converted_paths = convert_pytorch_to_multiple_formats(
                args.model_path,
                str(output_dir),
                pipeline_config
            )
        elif args.source_format == "tensorflow":
            converted_paths = convert_tensorflow_to_multiple_formats(
                args.model_path,
                str(output_dir),
                pipeline_config
            )
        elif args.source_format == "onnx":
            # Convert to optimized ONNX runtime
            converted_path = convert_onnx_to_optimized_runtime(
                args.model_path,
                str(output_dir),
                pipeline_config
            )
            converted_paths = [converted_path]
        
        # Convert to Safetensors if requested
        if "safetensors" in args.target_formats:
            safetensors_path = convert_to_safetensors(
                args.model_path,
                str(output_dir),
                args.source_format,
                pipeline_config
            )
            converted_paths.append(safetensors_path)
        
        # Print results
        print("\nConversion Results:")
        print("------------------")
        for path in converted_paths:
            print(f"Converted model saved to: {path}")
        
        # Print metadata for each converted model
        print("\nModel Metadata:")
        print("--------------")
        for path in converted_paths:
            converter = ModelConversionManager()
            metadata = converter.get_metadata(path)
            print(f"\nModel: {path}")
            for key, value in metadata.items():
                print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 