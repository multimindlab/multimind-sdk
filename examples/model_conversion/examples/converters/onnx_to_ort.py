#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import Dict, Any
import onnx
import onnxruntime
from multimind.model_conversion import ModelConversionManager

def convert_onnx_to_ort(
    model_path: str,
    output_path: str,
    config: Dict[str, Any]
) -> str:
    """Convert ONNX model to ONNX Runtime format with advanced options."""
    converter = ModelConversionManager(
        source_format="onnx",
        target_format="ort",
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
                            "level": config.get("optimization_level", "all"),
                            "providers": config.get("providers", ["CPUExecutionProvider"]),
                            "session_options": {
                                "graph_optimization_level": config.get("graph_optimization_level", "ORT_ENABLE_ALL"),
                                "enable_mem_pattern": config.get("enable_mem_pattern", True),
                                "enable_mem_reuse": config.get("enable_mem_reuse", True),
                                "execution_mode": config.get("execution_mode", "sequential")
                            }
                        }
                    }
                },
                {
                    "step": "convert",
                    "format_options": {
                        "ort": {
                            "save_as_external_data": config.get("save_as_external_data", False),
                            "external_data_threshold": config.get("external_data_threshold", 1024),
                            "custom_ops": config.get("custom_ops", {})
                        }
                    }
                }
            ]
        }
    )
    
    return converter.convert(model_path, output_path)

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to ONNX Runtime format")
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to ONNX model")
    parser.add_argument("--output-path", type=str, required=True,
                      help="Path to save ONNX Runtime model")
    parser.add_argument("--optimization-level", type=str, default="all",
                      choices=["basic", "extended", "all"],
                      help="Optimization level")
    parser.add_argument("--providers", type=str, nargs="+",
                      default=["CPUExecutionProvider"],
                      choices=["CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"],
                      help="Execution providers")
    parser.add_argument("--graph-optimization-level", type=str,
                      default="ORT_ENABLE_ALL",
                      choices=["ORT_DISABLE_ALL", "ORT_BASIC_OPT", "ORT_EXTENDED_OPT", "ORT_ENABLE_ALL"],
                      help="Graph optimization level")
    parser.add_argument("--execution-mode", type=str, default="sequential",
                      choices=["sequential", "parallel"],
                      help="Execution mode")
    parser.add_argument("--save-as-external-data", action="store_true",
                      help="Save large tensors as external data")
    
    args = parser.parse_args()
    
    config = {
        "optimization_level": args.optimization_level,
        "providers": args.providers,
        "graph_optimization_level": args.graph_optimization_level,
        "execution_mode": args.execution_mode,
        "save_as_external_data": args.save_as_external_data
    }
    
    try:
        output_path = convert_onnx_to_ort(
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