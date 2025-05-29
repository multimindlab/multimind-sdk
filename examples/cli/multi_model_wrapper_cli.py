import argparse
import logging
from model_wrapper import ModelWrapper

logger = logging.getLogger(__name__)

def main():
    # Initialize wrapper first to get available models
    wrapper = ModelWrapper()
    available = wrapper.available_models()
    
    if not available:
        print("No models available. Please check your API keys and Ollama installation.")
        return
    
    parser = argparse.ArgumentParser(description="Query various LLM models using CLI")
    parser.add_argument(
        "--model",
        choices=available,  # Use dynamically available models
        required=True,
        help=f"Available models: {', '.join(available)}"
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--ollama-model", type=str, default="mistral")
    parser.add_argument("--hf-model-id", type=str, default="mistralai/Mistral-7B-v0.1")
    
    args = parser.parse_args()
    
    logger.info(f"Querying {args.model} with prompt: {args.prompt}")
    
    result = wrapper.query_model(
        model=args.model,
        prompt=args.prompt,
        ollama_model=args.ollama_model,
        hf_model_id=args.hf_model_id
    )
    
    if result["status"] == "success":
        print(f"\n--- {args.model.upper()} Response ---\n")
        print(result["response"])
    else:
        print(f"\nError: {result['error']}")

if __name__ == "__main__":
    main() 