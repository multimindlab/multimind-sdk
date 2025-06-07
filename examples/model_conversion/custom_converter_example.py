from multimind.model_conversion import ModelConversionManager, ONNXConverter

def main():
    # Initialize the conversion manager
    manager = ModelConversionManager()
    
    # Create and register a custom ONNX converter with specific configuration
    onnx_converter = ONNXConverter()
    manager.register_converter("custom_onnx", onnx_converter)
    
    # List available converters (should now include our custom converter)
    print("Available converters:", manager.list_converters())
    
    # Example: Convert a model to ONNX format
    try:
        # Replace these paths with actual model paths
        model_path = "path/to/huggingface/model"
        output_path = "path/to/output/model"
        
        # Validate the model
        if manager.validate_model(model_path, "custom_onnx"):
            print("Model validation successful")
            
            # Get model metadata
            metadata = manager.get_model_metadata(model_path, "custom_onnx")
            print("Model metadata:", metadata)
            
            # Convert the model with custom ONNX configuration
            converted_path = manager.convert(
                model_path=model_path,
                output_path=output_path,
                converter_name="custom_onnx",
                config={
                    "opset_version": 13,
                    "device": "cuda",
                    "dynamic_axes": {
                        "input_ids": {0: "batch_size", 1: "sequence"},
                        "attention_mask": {0: "batch_size", 1: "sequence"},
                        "output": {0: "batch_size", 1: "sequence"}
                    },
                    "input_names": ["input_ids", "attention_mask"],
                    "output_names": ["output"]
                }
            )
            print(f"Model converted successfully to: {converted_path}")
            
            # Verify the converted model
            if manager.validate_model(converted_path, "custom_onnx"):
                print("Converted model validation successful")
                converted_metadata = manager.get_model_metadata(converted_path, "custom_onnx")
                print("Converted model metadata:", converted_metadata)
            else:
                print("Converted model validation failed")
        else:
            print("Model validation failed")
            
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    main() 