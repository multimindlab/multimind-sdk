#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from multimind.model_conversion import ModelConversionManager

def download_qwen_model(model_name: str, output_dir: str):
    """Download Qwen model from HuggingFace."""
    from huggingface_hub import snapshot_download
    
    print(f"Downloading {model_name} from HuggingFace...")
    model_path = snapshot_download(
        repo_id=model_name,
        local_dir=output_dir,
        local_dir_use_symlinks=False
    )
    print(f"Model downloaded to: {model_path}")
    return model_path

def convert_to_ollama(model_path: str, output_path: str, quantization: str = "q4_k_m"):
    """Convert model to Ollama GGUF format."""
    manager = ModelConversionManager()
    
    # Validate the model
    print("Validating source model...")
    if not manager.validate_model(model_path, "huggingface"):
        raise ValueError("Source model validation failed")
    print("Source model validation successful")
    
    # Get model metadata
    print("\nSource model metadata:")
    metadata = manager.get_model_metadata(model_path, "huggingface")
    for key, value in metadata.items():
        print(f"{key}: {value}")
    
    # Convert the model
    print(f"\nConverting model to Ollama GGUF format with {quantization} quantization...")
    converted_path = manager.convert(
        model_path=model_path,
        output_path=output_path,
        converter_name="huggingface",
        config={
            "format": "gguf",
            "quantization": quantization,
            "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        }
    )
    print(f"Model converted successfully to: {converted_path}")
    
    # Verify the converted model
    print("\nVerifying converted model...")
    if manager.validate_model(converted_path, "ollama"):
        print("Converted model validation successful")
        converted_metadata = manager.get_model_metadata(converted_path, "ollama")
        print("\nConverted model metadata:")
        for key, value in converted_metadata.items():
            print(f"{key}: {value}")
    else:
        print("Converted model validation failed")
    
    return converted_path

def test_converted_model(model_path: str):
    """Test the converted model using Ollama."""
    import requests
    import json
    
    # Prepare test prompt
    test_prompt = "Explain quantum computing in simple terms."
    
    print("\nTesting converted model with Ollama...")
    try:
        # Send request to Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": os.path.basename(model_path),
                "prompt": test_prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\nModel response:")
            print(result["response"])
            print(f"\nGeneration time: {result.get('total_duration', 'N/A')} seconds")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error testing model: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Convert Qwen model to Ollama GGUF format")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen1.5-7B",
                      help="HuggingFace model name (default: Qwen/Qwen1.5-7B)")
    parser.add_argument("--output-dir", type=str, default="./models",
                      help="Output directory for downloaded and converted models")
    parser.add_argument("--quantization", type=str, default="q4_k_m",
                      choices=["q4_k_m", "q4_0", "q5_k_m", "q8_0"],
                      help="Quantization method for GGUF format")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download model
        model_path = download_qwen_model(args.model_name, str(output_dir))
        
        # Convert model
        converted_path = convert_to_ollama(
            model_path=model_path,
            output_path=str(output_dir / "converted"),
            quantization=args.quantization
        )
        
        # Test converted model
        test_converted_model(converted_path)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 