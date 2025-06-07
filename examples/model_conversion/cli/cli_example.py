#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from multimind.model_conversion import ModelConversionManager

def parse_args():
    parser = argparse.ArgumentParser(description='Model Conversion CLI')
    
    # Required arguments
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the source model')
    parser.add_argument('--output-path', type=str, required=True,
                      help='Path where the converted model should be saved')
    parser.add_argument('--converter', type=str, required=True,
                      choices=['huggingface', 'ollama', 'custom_onnx'],
                      help='Converter to use for conversion')
    
    # Optional arguments
    parser.add_argument('--opset-version', type=int, default=12,
                      help='ONNX opset version (for ONNX conversion)')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cpu', 'cuda'],
                      help='Device to use for conversion')
    parser.add_argument('--quantization', type=str,
                      choices=['int8', 'int4', 'fp16'],
                      help='Quantization method to use')
    parser.add_argument('--validate', action='store_true',
                      help='Validate the model before and after conversion')
    parser.add_argument('--metadata', action='store_true',
                      help='Print model metadata')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize the conversion manager
    manager = ModelConversionManager()
    
    try:
        # Validate input paths
        model_path = Path(args.model_path)
        output_path = Path(args.output_path)
        
        if not model_path.exists():
            print(f"Error: Model path does not exist: {model_path}")
            sys.exit(1)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare conversion config
        config = {
            "device": args.device
        }
        
        if args.converter == "custom_onnx":
            config["opset_version"] = args.opset_version
        
        if args.quantization:
            config["quantization"] = args.quantization
        
        # Validate model if requested
        if args.validate:
            print("Validating source model...")
            if not manager.validate_model(str(model_path), args.converter):
                print("Error: Source model validation failed")
                sys.exit(1)
            print("Source model validation successful")
        
        # Get and print metadata if requested
        if args.metadata:
            print("\nSource model metadata:")
            metadata = manager.get_model_metadata(str(model_path), args.converter)
            for key, value in metadata.items():
                print(f"{key}: {value}")
        
        # Convert the model
        print(f"\nConverting model using {args.converter} converter...")
        converted_path = manager.convert(
            model_path=str(model_path),
            output_path=str(output_path),
            converter_name=args.converter,
            config=config
        )
        print(f"Model converted successfully to: {converted_path}")
        
        # Validate converted model if requested
        if args.validate:
            print("\nValidating converted model...")
            if not manager.validate_model(converted_path, args.converter):
                print("Error: Converted model validation failed")
                sys.exit(1)
            print("Converted model validation successful")
        
        # Get and print converted model metadata if requested
        if args.metadata:
            print("\nConverted model metadata:")
            converted_metadata = manager.get_model_metadata(converted_path, args.converter)
            for key, value in converted_metadata.items():
                print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 