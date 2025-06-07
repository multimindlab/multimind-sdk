#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import Dict, Any
import tensorflow as tf
from multimind.model_conversion import ModelConversionManager

def convert_tensorflow_to_tflite(
    model_path: str,
    output_path: str,
    config: Dict[str, Any]
) -> str:
    """Convert TensorFlow model to TFLite format with advanced options."""
    converter = ModelConversionManager(
        source_format="tensorflow",
        target_format="tflite",
        pipeline_config={
            "pipeline": [
                {
                    "step": "validate",
                    "checks": ["format", "metadata"]
                },
                {
                    "step": "optimize",
                    "methods": ["quantization"],
                    "config": {
                        "quantization": {
                            "method": config.get("quantization", "dynamic"),
                            "calibration_data": config.get("calibration_data"),
                            "optimizations": config.get("optimizations", ["DEFAULT"]),
                            "target_spec": {
                                "supported_ops": config.get("supported_ops", ["TFLITE_BUILTINS"]),
                                "supported_types": config.get("supported_types", ["FLOAT"]),
                                "select_user_tf_ops": config.get("select_user_tf_ops", [])
                            }
                        }
                    }
                },
                {
                    "step": "convert",
                    "format_options": {
                        "tflite": {
                            "allow_custom_ops": config.get("allow_custom_ops", False),
                            "experimental_new_converter": config.get("experimental_new_converter", True),
                            "experimental_new_quantizer": config.get("experimental_new_quantizer", True)
                        }
                    }
                }
            ]
        }
    )
    
    return converter.convert(model_path, output_path)

def main():
    parser = argparse.ArgumentParser(description="Convert TensorFlow model to TFLite format")
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to TensorFlow model")
    parser.add_argument("--output-path", type=str, required=True,
                      help="Path to save TFLite model")
    parser.add_argument("--quantization", type=str, default="dynamic",
                      choices=["dynamic", "float16", "int8"],
                      help="Quantization method")
    parser.add_argument("--optimizations", type=str, nargs="+",
                      default=["DEFAULT"],
                      choices=["DEFAULT", "OPTIMIZE_FOR_LATENCY", "OPTIMIZE_FOR_SIZE"],
                      help="TFLite optimizations")
    parser.add_argument("--supported-ops", type=str, nargs="+",
                      default=["TFLITE_BUILTINS"],
                      choices=["TFLITE_BUILTINS", "TFLITE_BUILTINS_INT8", "SELECT_TF_OPS"],
                      help="Supported operations")
    parser.add_argument("--allow-custom-ops", action="store_true",
                      help="Allow custom operations")
    
    args = parser.parse_args()
    
    config = {
        "quantization": args.quantization,
        "optimizations": args.optimizations,
        "supported_ops": args.supported_ops,
        "allow_custom_ops": args.allow_custom_ops
    }
    
    try:
        output_path = convert_tensorflow_to_tflite(
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