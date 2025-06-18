#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List
from multimind.model_conversion import ModelConversionManager

def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="MultiMind SDK Model Conversion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert HuggingFace model to GGUF
  multimind convert --source huggingface --target gguf --model-path Qwen/Qwen1.5-7B --output-dir ./models

  # Convert PyTorch model to Safetensors with compression
  multimind convert --source pytorch --target safetensors --model-path ./model.pt --compression lz4

  # Convert TensorFlow model to TFLite with optimization
  multimind convert --source tensorflow --target tflite --model-path ./model --optimizations DEFAULT

  # Convert ONNX model to ONNX Runtime
  multimind convert --source onnx --target ort --model-path ./model.onnx --optimization-level all
        """
    )

    # Required arguments
    parser.add_argument("--source", type=str, required=True,
                      choices=["huggingface", "pytorch", "tensorflow", "onnx", "ollama"],
                      help="Source model format")
    parser.add_argument("--target", type=str, required=True,
                      choices=["gguf", "safetensors", "tflite", "ort", "onnx"],
                      help="Target model format")
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to source model or HuggingFace model ID")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Directory to save converted model")

    # Optional arguments
    parser.add_argument("--quantization", type=str,
                      choices=["q4_k_m", "q4_0", "q5_k_m", "q8_0", "int8", "fp16"],
                      help="Quantization method")
    parser.add_argument("--compression", type=str,
                      choices=["lz4", "zstd"],
                      help="Compression method for Safetensors")
    parser.add_argument("--compression-level", type=int, default=9,
                      help="Compression level (1-9)")
    parser.add_argument("--optimizations", type=str, nargs="+",
                      help="Optimization methods (e.g., DEFAULT OPTIMIZE_FOR_LATENCY)")
    parser.add_argument("--optimization-level", type=str,
                      choices=["basic", "all", "extreme"],
                      help="Optimization level for ONNX Runtime")
    parser.add_argument("--device", type=str, default="cpu",
                      choices=["cpu", "cuda"],
                      help="Device to use for conversion")
    parser.add_argument("--context-length", type=int,
                      help="Context length for GGUF models")
    parser.add_argument("--metadata", type=str, nargs="+",
                      help="Additional metadata (key=value pairs)")
    parser.add_argument("--validate", action="store_true",
                      help="Validate model before and after conversion")
    parser.add_argument("--test", action="store_true",
                      help="Test converted model")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output")

    return parser

def parse_metadata(metadata_args: List[str]) -> Dict[str, str]:
    """Parse metadata arguments into dictionary."""
    if not metadata_args:
        return {}
    return dict(pair.split("=") for pair in metadata_args)

def get_conversion_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Generate conversion configuration from arguments."""
    config = {
        "device": args.device
    }

    # Add format-specific configurations
    if args.quantization:
        config["quantization"] = args.quantization
    if args.compression:
        config["compression"] = {
            "method": args.compression,
            "level": args.compression_level
        }
    if args.optimizations:
        config["optimizations"] = args.optimizations
    if args.optimization_level:
        config["optimization_level"] = args.optimization_level
    if args.context_length:
        config["context_length"] = args.context_length
    if args.metadata:
        config["metadata"] = parse_metadata(args.metadata)

    return config

def validate_model(manager: ModelConversionManager, model_path: str, format: str) -> bool:
    """Validate model format."""
    try:
        if manager.validate_model(model_path, format):
            print(f"✓ {format.upper()} model validation successful")
            return True
        else:
            print(f"✗ {format.upper()} model validation failed")
            return False
    except Exception as e:
        print(f"✗ Error validating {format.upper()} model: {str(e)}")
        return False

def print_metadata(metadata: Dict[str, Any]):
    """Print model metadata."""
    print("\nModel Metadata:")
    print("--------------")
    for key, value in metadata.items():
        print(f"{key}: {value}")

def main():
    parser = setup_parser()
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize conversion manager
    manager = ModelConversionManager()

    try:
        # Validate source model if requested
        if args.validate:
            print("\nValidating source model...")
            if not validate_model(manager, args.model_path, args.source):
                return 1

        # Get source model metadata
        if args.verbose:
            print("\nSource model metadata:")
            metadata = manager.get_model_metadata(args.model_path, args.source)
            print_metadata(metadata)

        # Prepare conversion configuration
        config = get_conversion_config(args)

        # Convert model
        print(f"\nConverting {args.source.upper()} model to {args.target.upper()} format...")
        converted_path = manager.convert(
            model_path=args.model_path,
            output_path=str(output_dir),
            converter_name=args.source,
            config=config
        )
        print(f"✓ Model converted successfully to: {converted_path}")

        # Validate converted model if requested
        if args.validate:
            print("\nValidating converted model...")
            if not validate_model(manager, converted_path, args.target):
                return 1

        # Get converted model metadata
        if args.verbose:
            print("\nConverted model metadata:")
            metadata = manager.get_model_metadata(converted_path, args.target)
            print_metadata(metadata)

        # Test converted model if requested
        if args.test:
            print("\nTesting converted model...")
            if args.target == "gguf":
                from examples.model_conversion.examples.qwen_to_ollama import test_converted_model
                test_converted_model(converted_path)
            else:
                print("Model testing not implemented for this format")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 