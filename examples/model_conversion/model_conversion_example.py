from multimind.model_conversion import ModelConversionManager

def main():
    # Initialize the conversion manager
    manager = ModelConversionManager()
    
    # List available converters
    print("Available converters:", manager.list_converters())
    
    # Example: Convert a HuggingFace model to Ollama format
    try:
        # Replace these paths with actual model paths
        model_path = "path/to/huggingface/model"
        output_path = "path/to/output/model"
        
        # Validate the model
        if manager.validate_model(model_path, "huggingface"):
            print("Model validation successful")
            
            # Get model metadata
            metadata = manager.get_model_metadata(model_path, "huggingface")
            print("Model metadata:", metadata)
            
            # Convert the model
            converted_path = manager.convert(
                model_path=model_path,
                output_path=output_path,
                converter_name="huggingface",
                config={
                    "quantization": "int8",
                    "device": "cuda"
                }
            )
            print(f"Model converted successfully to: {converted_path}")
        else:
            print("Model validation failed")
            
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    main() 